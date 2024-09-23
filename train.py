# from modelings.modeling_chatglm import ChatGLMForConditionalGeneration
# from tokenizers.tokenization_chatglm import ChatGLMTokenizer
# from configs.configuration_chatglm import ChatGLMConfig

import torch
import deepspeed
from tqdm import tqdm
from models.msp import LM4MTModel
from models.osp import OLM4MTModel
import opts
from inputters.dataset import LM4MTDataset, DataCollator
from torch.utils.data import DataLoader
import utils.utils as utils
import os
import shutil
import json
from shutil import copy
from llm_loader import MODE
import math
import configargparse

from torch.utils.data import RandomSampler, DistributedSampler, DataLoader


import logging
logger = logging.getLogger()



try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter



def cover_ds_params(deepspeedconfig, args):

    deepspeedconfig['train_micro_batch_size_per_gpu'] = 1
    deepspeedconfig['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    
    deepspeedconfig["optimizer"]["params"]["lr"] = args.learning_rate
    deepspeedconfig["optimizer"]["params"]["betas"] = (0.9, 0.95)
    deepspeedconfig["optimizer"]["params"]["eps"] = 1e-8
    # deepspeedconfig["optimizer"]["params"]["weight_decay"] = 0.1
    
    deepspeedconfig["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    deepspeedconfig["scheduler"]["params"]["warmup_min_lr"] = 0
    deepspeedconfig["scheduler"]["params"]["total_num_steps"] = args.num_training_steps
    deepspeedconfig["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps

def main(args):
    
    # distribution initialation
    
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()
    
    if args.global_rank <= 0:
        tb_write = SummaryWriter()

    utils.set_random_seed(args.seed)
    torch.distributed.barrier()
    
    # deepspeed params cover
    with open(args.ds_file, "r", encoding="utf-8") as fh:
        ds_config = json.load(fh)
    
    
    # load tokenizer
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.model_dir)
    utils.print_rank_0("tokenizer.bos_token: {}".format(tokenizer.bos_token), args.global_rank)
    utils.print_rank_0("tokenizer.pad_token: {}".format(tokenizer.pad_token), args.global_rank)
    utils.print_rank_0("tokenizer.eos_token: {}".format(tokenizer.eos_token), args.global_rank)

    # load the pretrainied large language model
    print("Loading Pre-trained language model...", flush=True)
    model = MODE[args.mode]["model"].from_pretrained(args.model_dir)
    model.add_new_params(args)
   
    if args.data_type == "msp":
        model = LM4MTModel(model, args)
    else:
        model = OLM4MTModel(model, args)
    
    print(model)
    print("Finished.", flush=True)
    # fix the pretrained-model params
    for name, param in model.lm_model.named_parameters():
        if "q_scaling_product" not in name and "source_info_proj" not in name:
            param.requires_grad = False
        
       
    utils.print_trainable_parameters(model)

    # init the dataloader
    
    train_dataset = LM4MTDataset(args.train_path, tokenizer, args.max_src_len, args.max_tgt_len, batch_size=args.train_batch_size, workers_num=16, ctx_len_multiplier=args.ctx_sent_num)
   
    data_collator = DataCollator(tokenizer=tokenizer, is_train=True)
    
    single_epoach_step = math.ceil(len(train_dataset) / (args.gradient_accumulation_steps * torch.distributed.get_world_size()))
    train_steps = args.num_train_epochs * single_epoach_step
    
    if args.num_training_steps > train_steps:
        args.num_train_epochs = math.ceil(args.num_training_steps / single_epoach_step)
    else:
        args.num_training_steps = train_steps
    
    utils.print_rank_0("num_training_steps = {}".format(args.num_training_steps), args.global_rank)
    args.num_warmup_steps = int(args.warmup_ratio * args.num_training_steps)
    utils.print_rank_0("num_warmup_steps = {}".format(args.num_warmup_steps), args.global_rank)
    
    cover_ds_params(ds_config, args)
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(config=ds_config,
                                                         model=model, training_data=train_dataset, collate_fn=data_collator, dist_init_required=True)
    
    if args.recover_training_from:
        logger.info("INFO: Recover the training from {}".format(args.recover_training_from))
        model_engine.load_checkpoint(load_dir=args.recover_training_from)
    
    model_engine.train()
    
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    
    for i_epoch in range(args.num_train_epochs):
        utils.print_rank_0("Beginning of Epoch {}/{}, Total Micro Batches {}".format(i_epoch + 1, args.num_train_epochs,
                                                                               len(train_dataloader)), args.global_rank)
        
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            inputs, labels = batch
            inputs = utils.to_device(inputs, device=device)
            labels.to(device)
            loss = model_engine.forward(features=inputs, labels=labels)
            tr_loss += loss.item()
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model_engine.parameters(), 1.0)
            model_engine.step()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # write loss
                if global_step % args.log_steps == 0:
                    utils.print_rank_0("Epoch: {}, step: {}, global_step:{}, loss: {}".format(i_epoch, step + 1, global_step, 
                                                                                              (tr_loss - logging_loss)/(args.log_steps * args.gradient_accumulation_steps)), args.global_rank)
                    utils.print_rank_0("step: {}-{}-{}".format(step + 1, global_step, model_engine.global_steps), args.global_rank)
                    if args.global_rank <= 0:
                        tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                            (args.log_steps * args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss
                # save model
                if args.save_steps is not None and global_step % args.save_steps == 0:
                    lastest_step = global_step - args.save_steps
                    train_lastest_ckpt_dir = "global_step" + str(lastest_step)
                    
                    if lastest_step > 0 and args.global_rank == 0:
                        rm_dir = args.output_dir + "/" + train_lastest_ckpt_dir
                        if os.path.exists(rm_dir):
                            shutil.rmtree(args.output_dir + "/" + train_lastest_ckpt_dir)
                    
                    model_engine.save_checkpoint(save_dir=args.output_dir)
                    
                    if ds_config["zero_optimization"]["stage"] == 3:
                        state_dict = model_engine._zero3_consolidated_16bit_state_dict()
                        if args.global_rank <= 0:
                            utils.save_checkpoint(global_step, i_epoch, state_dict, optimizer, args)
                    else:
                        if args.global_rank <= 0:
                            utils.save_checkpoint(global_step, i_epoch, model.state_dict(), optimizer, args)

                    print("saving the step {} checkpoint to {}".format(global_step, args.output_dir))
                    
                    model_engine.eval()
                    
                    print("Evaluating the loss on Dev set...")
                    dev_loss = 0
                    with torch.no_grad():
                        dev_dataset = LM4MTDataset(args.dev_path, tokenizer, args.max_src_len, args.max_tgt_len, batch_size=args.train_batch_size, workers_num=2, ctx_len_multiplier=args.ctx_sent_num)
                        if args.local_rank == -1:
                            dev_sampler = RandomSampler(dev_dataset)
                        else:
                            dev_sampler = DistributedSampler(dev_dataset, shuffle=True)
                
                        data_collator = DataCollator(tokenizer=tokenizer, is_train=True)
                        dev_dataloader = DataLoader(train_dataset, sampler=dev_sampler, collate_fn=data_collator, num_workers=8)
                        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), unit="batch"):
                            inputs, labels = batch
                            inputs = utils.to_device(inputs, device=device)
                            labels.to(device)
                            loss = model_engine.forward(features=inputs, labels=labels)
                            dev_loss += loss.item()
                        
                        print("the dev loss: {}".format(dev_loss / len(dev_dataset)))
                    
                    model_engine.train()   
        
        if ds_config["zero_optimization"]["stage"] == 3:
            state_dict = model_engine._zero3_consolidated_16bit_state_dict()
            if args.global_rank <= 0:
                utils.save_checkpoint(global_step, i_epoch, state_dict, optimizer, args)
        else:
            if args.global_rank <= 0:
                utils.save_checkpoint(global_step, i_epoch, model.state_dict(), optimizer, args)
        
        model_engine.train()
        print("saving the step {} checkpoint to {}".format(global_step, args.output_dir))

    print("the training has beeb done, and the model is trained for {} steps".format(global_step))


if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
    description='train.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  
  parser = deepspeed.add_config_arguments(parser)
  opts.training_opts(parser)
  
  opts.model_opts(parser)
  opt = parser.parse_args()
  main(opt)