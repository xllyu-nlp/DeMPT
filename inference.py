# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import copy
import re
import socket
import time
import opts
import torch
import configargparse
import torch.distributed as dist
import utils.decoding_strategy as ds
from transformers import LlamaForCausalLM, LlamaTokenizer
from inputters.dataset import LM4MTDataset, DataCollator, infer_input_fn, to_translation_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from models.msp import LM4MTModel
from models.osp import OLM4MTModel
import logging
import utils.utils as utils
from llm_loader import MODE
logger = logging.getLogger()

def convert_to_string(tensor, tokenizer):
    ids = tensor.tolist()
    s = tokenizer.decode(ids)

    idx = s.find("</s>")

    if idx != -1:
        s = s[:idx]

    idx = s.find("<pad>")

    if idx != -1:
        s = s[:idx]
    s = s.strip().encode("utf-8")
    
    return s


def main(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(args.device_list))
    torch.cuda.set_device(args.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

   
    # Create model
    with torch.no_grad():
        # Load configs
        tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.ptm)
        model = MODE[args.mode]["model"].from_pretrained(args.ptm)
        model.add_new_params(args)
       
        args.bos_id = model.config.bos_token_id
        args.eos_id = model.config.eos_token_id
        args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        args.sep_id = model.config.sep_token_id if model.config.sep_token_id else model.config.bos_token_id
        
        args.tokenizer = tokenizer
        vocab_size = tokenizer.vocab_size
        
        
        model.to(device)
        model.eval()
        
        print("Finished.", flush=True)
        if args.data_type == "msp":
            model = LM4MTModel(model, args, is_train=False)
        else:
            model = OLM4MTModel(model, args, is_train=False)
        print(model)
        model.load_prefix(args.prefix)
        
        model = model.to(device)

        model.eval()
        
        # buliding the inference datasets
       
        sorted_key, dataset= infer_input_fn(args)
        
        dataloader = iter(dataset)
        counter = 0
        pad_max = 1024
        top_beams = args.top_beams
        decode_batch_size = args.batch_size

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([decode_batch_size, top_beams, pad_max]).long()
                  for _ in range(dist.get_world_size())]
        
        all_outputs = []
        
        
        while True:
            try:
                features = next(dataloader)
                features = to_translation_features(features, args.pad_id)
                
                batch_size = features["source"].shape[0]
            except Exception as e:
                features = {
                    "source": torch.ones([1, 1]).long() * 1,
                    "source_mask": torch.ones([1, 1]).float(),
                    "context": torch.ones([1, 1]).long() * 1,
                    "context_mask": torch.ones([1, 1]).float()
                }
                batch_size = 0
            
            features = utils.to_device(features, device)
            t = time.time()
            counter += 1
        
            seqs, _ = ds.beam_search([model], features, args)
            
            # Padding
            pad_batch = decode_batch_size - seqs.shape[0]
            pad_beams = top_beams - seqs.shape[1]
            pad_length = pad_max - seqs.shape[2]
            seqs = torch.nn.functional.pad(
                seqs, (0, pad_length, 0, pad_beams, 0, pad_batch))
            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)
            
            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(decode_batch_size):
                for j in range(dist.get_world_size()):
                    beam_seqs = []
                    
                    pad_flag = i >= size[j]

                    for k in range(top_beams):
                        seq = convert_to_string(t_list[j][i][k], tokenizer)
                    
                        if pad_flag:
                            continue
            
                        beam_seqs.append(seq)
                        
                    if pad_flag:
                        continue
                    
                    all_outputs.append(beam_seqs)
                   
            t = time.time() - t
            print("Finished batch: %d (%.3f sec)" % (counter, t))
            
        
        if dist.get_rank() == 0:
            restored_outputs = [] 
            if sorted_key is not None:
                for idx in range(len(all_outputs)):
                    restored_outputs.append(all_outputs[sorted_key[idx]])
            else:
                restored_outputs = all_outputs

            with open(args.output, "wb") as fd:
                if top_beams == 1:
                    for seqs in restored_outputs:
                        fd.write(seqs[0] + b"\n")
                else:
                    for idx, seqs in enumerate(restored_outputs):
                        for k, seq in enumerate(seqs):
                            fd.write(b"%d\t%d\t" % (idx, k))
                            fd.write(seq + b"\n")

# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)

if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
    description='inference.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

  
  
  opts.model_opts(parser)
  opts.inference_opts(parser)
  opt = parser.parse_args()
  with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        opt.url = url

  world_size = len(opt.device_list)

  if world_size > 1:
    torch.multiprocessing.spawn(process_fn, args=(opt,),
                                  nprocs=world_size)
  else:
    process_fn(0, opt)