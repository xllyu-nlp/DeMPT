from __future__ import print_function



def training_opts(parser):
    group = parser.add_argument_group("Training-Settings")
    # Dataset
    group.add("--train_path", "-train_path", type=str, required=True, 
              help="path to training file")
    
    group.add("--dev_path", "-dev_path", type=str, required=True, 
              help="path to dev file")

    group.add("--data_type", "-data_type", type=str, default="msp",
                        help="Path to input file")

   # Trainging features
    group.add("--train_batch_size", "-train_batch_size", type=int, default=1024, 
              help="max number of tokens in a mini-batch")
    
    group.add("--num_train_epochs", "-num_train_epochs", type=int, default=20, 
              help="max number of epoches")
    group.add("--num_training_steps", "-num_training_steps", type=int, default=8000,
              help="max number of epoches")
    group.add("--num_warmup_steps", "-num_warmup_steps", type=int, default=5000,
              help="max number of epoches")

    group.add("--gradient_accumulation_steps", "-gradient_accumulation_steps", type=int, default=1, 
              help="the number of graddient accumulation")
    group.add("--output_dir", "-output_dir", type=str, required=True, 
              help="path to storing the model, log")
    
    group.add("--recover_training_from", "-recover_training_from", type=str, default="", 
              help="recover training from the dir")

    group.add("--weight_decay", "-weight_decay", type=float, default=0.1, help="")
    group.add("--warmup_ratio", "-warmup_ratio", type=float, default=0.1, 
              help="the warm up ratio for training")
    group.add("--mode", "-mode", type=str, default="llama", help="the pretrained model mnode")
    group.add("--seed", "-seed", type=int, default=1234, help="")
    group.add("--model_dir", "-model_dir", type=str, required=True, 
              help="path to pretrained language model")
    
    group.add("--log_steps", "-log_steps", type=int, default=20,
              help="number of steps for reporting the training process")
    group.add("--save_steps", "-save_steps", type=int, default=1000,
              help="number of steps for saving the training model")
    group.add("--keep_checkpoint_max", "-keep_checkpoint_max", type=int, default=20,
              help="max number of saved checkpoint during whole the training")
    
    group.add("--dev_steps", "-dev_steps", type=int, default=20,
              help="number of steps for reporting the training process")
    

    group.add("--learning_rate", "-learning_rate", type=float, default=1e-3, help="")
    
    group.add("--use_ctx", action="store_true",
                        help="Use half precision for decoding")

    group.add("--pre_seq_len", "-pre_sef_len", type=int, default=64, 
              help="the length of prompt text")
    group.add("--local_rank", "-local_rank", type=int, default=-1, 
              help="the gpu ranking")
    group.add("--prompt_text", "-prompt_text", type=str, default="你是一个翻译模型，请翻译如下的句子到英文：", 
              help="the prompt text for llm")
    group.add("--head_tuning", action="store_true",
                        help="if tuning the head layer of llm")
    # Deepspeed setting
    group.add("--ds_file", "-ds_file", type=str, default="./ds_configs/zero_stage2.json",
              help="config file for deepspeed")

    

def inference_opts(parser):
    # input files
    group = parser.add_argument_group("Inference-Settings")

    group.add("--input", "-input", type=str, required=True,
                        help="Path to input file")
    group.add("--data_type", "-data_type", type=str, default="msp",
                        help="Path to input file")
    group.add("--use_ctx", action="store_true",
                        help="Use half precision for decoding")


    group.add("--output", "-output", type=str, required=True,
                        help="Path to output file")
    group.add("--ptm", "-ptm", type=str, required=True,
                        help="Path to pre-trained checkpoint")
    group.add("--prefix", "-prefix", type=str, required=True,
                        help="Path to prefix parameters")
    group.add("--half", action="store_true",
                        help="Use half precision for decoding")
    group.add("--mode", "-mode", type=str, default="llama", help="the pretrained model mode")
    
    # decoding params
    group = parser.add_argument_group("Decoding-Params")
    # vocabulary specific
    group.add("--pad", "-pad", type=str, default="<pad>", 
                        help="the padding token for dataset")
    group.add("--bos", "-bos", type=str, default="<bos>", 
                        help="the BOS token for dataset")
    group.add("--eos", "-eos", type=str, default="<eos>", 
                        help="the EOS token for dataset")
    group.add("--unk", "-unk", type=str, default="<unk>", 
                        help="the unknown token for dataset")
    group.add('--device_list', '-device_list', default=[0], nargs='*', type=int,
                        help="the GPUs list")
    
    # decoding
    group.add("--top_beams", "-top_beams", type=int, default=1, 
                        help="the grady search for decoding")
    group.add("--beam_size", "-beam_size", type=int, default=4,
                        help="the size of beam for beam-search decoding")
    group.add("--decode_alpha", "-decode_alpha", type=float, default=0.6, 
                        help="the penalty ratio for decoding length")
    group.add("--decode_ratio", "-decode_ratio", type=float, default=1.0, 
                        help="the ratio of targe against with source")
    group.add("--decode_length", "-decode_length", type=int, default=50, 
                        help="the unknown token for dataset")
    group.add("--batch_size", "-batch_size", type=int, default=16,
                        help="the batch size for decoding ")


def model_opts(parser):
    # for prompt tuning param settings
    group = parser.add_argument_group("Prompt-Settings")
    group.add("--prompt_length", "-prompt_length", type=int, default=128)
    group.add("--dec_no_prefix", "-dec_no_prefix", action='store_true')
    group.add("--enhance_decode", "-enhance_decode", action='store_true')
    group.add("--share_prompt", "-share_prompt", action='store_true')
    group.add("--prompts_proj", "-prompts_proj", action='store_true')
    group.add("--enhance_proj", "-enhance_proj", action='store_true')
    group.add("--enhance_attn", "-enhance_attn", action='store_true')
    group.add("--source_annealing", "-source_annealing", action='store_true')
    group.add("--re_encoding", "-re_encoding", type=int, default=1)
    group.add("--label_smoothing", "-label_smoothing", type=float, default=0.1, 
            help="ratio of label smoothing")
   
    group.add("--keys_type_num", "-keys_type_num", type=int, default=0, 
            help="distinguish the different prompt and sequence")

    group.add("--share_proj_layer", action="store_true",
                        help="Use half precision for decoding")

    group.add("--ctx_sent_num", "-ctx_sent_num", type=int, default=1, 
            help="ratio of label smoothing")
    
    group.add("--max_tgt_len", "-max_len", type=int, default=1024,
              help="the max length of input")
    
    group.add("--max_src_len", "-max_src_len", type=int, default=1024)
    

