
from torch.utils.data import Dataset
import torch
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence

import math
import torch
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Manager, Pool


class LM4MTDataset(Dataset):

    def __init__(self, data_path, tokenizer, src_len, tgt_len, split_tok="<extra_id_0>", is_train=True, batch_size=512, workers_num=1, ctx_len_multiplier=1):
    
        lines = [line for line in open(data_path, "r", encoding="utf-8")]
        
        self.all_data = []
        self.batch_size = batch_size
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.ctx_num = ctx_len_multiplier
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.ctx_len = src_len * ctx_len_multiplier
        self.split_tok = split_tok
        
      
        split_count = math.ceil(len(lines) / workers_num)
        count = 0

        group_lines = []
        res = []
        for _ in range(workers_num):
            group_lines.append(lines[count: count+split_count])
            count += split_count
        
        with Pool(processes=workers_num) as p:
            res = list(p.imap(self.preprocess_inst, group_lines))
        
        for r in res:
            self.all_data.extend(r)
        
        self.create_batches()
        
    


    def preprocess_inst(self, lines):
        
        all_data = []
        for line in tqdm(lines):
            # print(line)
            
            sampler = line.strip().split(self.split_tok)    
            ctx, src = sampler[0], sampler[1]
            ctx_tokens, src_tokens = self.tokenizer.tokenize(ctx.strip()), self.tokenizer.tokenize(src.strip())
            # truncate the inputs for src, tgt and ctx, respectively.
            
            if len(ctx_tokens) > self.ctx_len or len(src_tokens) > self.src_len:
                continue
            
            # convert the tokens to ids
            ctx_tokens_ids = self.tokenizer.convert_tokens_to_ids(ctx_tokens)
            src_tokens_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
            if self.is_train:
                tgt = sampler[2]
                tgt_tokens = self.tokenizer.tokenize(tgt.strip())
                if len(tgt_tokens) > self.tgt_len:
                    continue
                # tgt_tokens = tgt_tokens[:self.tgt_len] if len(tgt_tokens) > self.tgt_len else tgt_tokens
                tgt_tokens = [self.tokenizer.bos_token] + tgt_tokens + [self.tokenizer.eos_token]
                tgt_tokens_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)   
                # instance_len = max(len(src_tokens_ids), len(tgt_tokens_ids))
                instance_len = round((len(ctx_tokens_ids) + len(src_tokens_ids) + len(tgt_tokens_ids)) / (self.ctx_num + 2))
                all_data.append({"ctx": ctx_tokens_ids, "src": src_tokens_ids, "tgt": tgt_tokens_ids, "input_len": instance_len})
            else:
                instance_len = len(src_tokens_ids)
                all_data.append({"ctx": ctx_tokens_ids, "src": src_tokens_ids, "input_len": instance_len})
        return all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance
    


    def create_batches(self):
        
        if self.is_train:
            self.all_data.sort(key=lambda x: x["input_len"])
            # Group or chunk based on target sequence lengths
            chunks = [list(g) for _, g in groupby(self.all_data, key=lambda x: x["input_len"])]

            # Create batches, each with the same target sequence length
            self.all_batches = list()
            
            for chunk in tqdm(chunks, desc="batching the data"):
                # Sort inside chunk by source sequence lengths, so that a batch would also have similar source sequence lengths
                chunk.sort(key=lambda x: x["input_len"])
                # How many sequences in each batch? Divide expected batch size (i.e. tokens) by target sequence length in this chunk
                seqs_per_batch = self.batch_size // chunk[0]["input_len"]
                # each batch must have at least 1 sequence
                seqs_per_batch = 1 if seqs_per_batch < 1 else seqs_per_batch
                # Split chunk into batches
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            # Shuffle batches
            self.n_batches = len(self.all_batches)
            self.all_data = self.all_batches
            
        else:
            # Simply return once pair at a time
            line_idx = [(i, inst["input_len"]) for i, inst in enumerate(self.all_data)]
            sorted_input_lens = sorted(line_idx, key=lambda x: x[1], reverse=True)
            sorted_keys = {}
            sorted_inputs = []
            
            for i, (idx, _) in enumerate(sorted_input_lens):
                sorted_inputs.append(self.all_data[idx])
                sorted_keys[idx] = i
            self.sorted_keys = sorted_keys
            self.sorted_inputs = sorted_inputs
            self.all_batches = [(self.sorted_inputs[i: i + self.batch_size], [i for i in range(i, i+self.batch_size)]) for i in range(0, len(self.sorted_inputs), self.batch_size)]
            self.n_batches = len(self.all_batches)
            self.all_data = self.all_batches
            
            
                
class DataCollator(object):
    
    def __init__(self, tokenizer, is_train=True):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        self.is_train = is_train
    
    def __call__(self, batch):
        
        try:
            # source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
            source_data, target_data, ctx_data, input_max_len = [], [], [], []
            if not self.is_train:
              inst_idx = batch[0][1]
              batch = [batch[0][0]]
            else:
              inst_idx = []
            

            for inst in batch[0]:
              ctx_data.append(inst['ctx'])
              source_data.append(inst['src'])
              input_max_len.append(inst['input_len'])
              if self.is_train:
                  target_data.append(inst['tgt'])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data],
                                   batch_first=True,
                                   padding_value=self.pad_id)
        
        ctx_data = pad_sequence(sequences=[torch.LongTensor(c) for c in ctx_data],
                                batch_first=True,
                                padding_value=self.pad_id)
        source_mask = (source_data != self.pad_id).long()
        ctx_mask = (ctx_data != self.pad_id).long()
        
        
        if self.is_train:
            target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data],
                                   batch_first=True,
                                   padding_value=self.pad_id)
            
            target_mask = (target_data != self.pad_id).long()
            
            labels = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data],
                                   batch_first=True,
                                   padding_value=-100)
            
            return {"source": torch.LongTensor(source_data),
                    "target": torch.LongTensor(target_data[:, :-1]),
                    "context": torch.LongTensor(ctx_data),
                    "source_mask": torch.LongTensor(source_mask),
                    "target_mask": torch.LongTensor(target_mask[:, :-1]),
                    "context_mask": torch.LongTensor(ctx_mask)}, torch.LongTensor(labels[:, 1:])
        else:
            return {"source": torch.LongTensor(source_data),
                    "context": torch.LongTensor(ctx_data),
                    "source_mask": torch.LongTensor(source_mask),
                    "context_mask": torch.LongTensor(ctx_mask)}, inst_idx



def sort_input_file(filename, reverse=True):
    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, sorted_inputs


def get_masks(x, pad_id):

    masks = []
    for ids in x:
        mask = []

        for id in ids:
            mask.append(1) if id != pad_id else mask.append(0)
            
        masks.append(mask)
    
    return masks



def infer_input_fn(args, sep_token="<extra_id_0>"):
    sorted_key, sorted_data = sort_input_file(args.input)
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(sorted_data))
    dataset = dataset.shard(torch.distributed.get_world_size(),
                            torch.distributed.get_rank())
    # args = get_args()
    tokenizer = args.tokenizer
    # data_type = args.data_type
    
    ctx_max_len = args.max_src_len * args.ctx_sent_num
    
    def py_tokenize(x):
        x = x.numpy().decode("utf-8", errors="ignore")
        sample = x.split(sep_token)
        ctx, src = sample[0], sample[1]
        
        ctx = tokenizer.encode(ctx.strip(), add_special_tokens=False)
        ctx = ctx if len(ctx) < ctx_max_len else ctx[len(ctx) - ctx_max_len:]
        src = tokenizer.encode(src.strip(), add_special_tokens=False)
        
        ctx = tf.convert_to_tensor(ctx, dtype=tf.int32)
        src = tf.convert_to_tensor(src, dtype=tf.int32)
        source_length = tf.shape(src)[0]
        
        return src, ctx, source_length

    def map_func(x):
        return tf.py_function(py_tokenize, [x], [tf.int32, tf.int32, tf.int32])

    dataset = dataset.map(map_func)
    
    dataset = dataset.map(lambda x, y, z: {"source": x, "context": y, "source_length": z}, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(
        args.batch_size,
         padded_shapes={
            "source": tf.TensorShape([None]),
            "context": tf.TensorShape([None]),
            "source_length": tf.TensorShape([])
        },
        padding_values={
            "source": args.pad_id,
            "context": args.pad_id,
            "source_length": 0
        })
    
    return sorted_key, dataset


def to_translation_features(features, pad_id):
    
    
    sources = features["source"]
    contexts = features["context"]
    lengths = features["source_length"]

    sources = sources.numpy().tolist()
    contexts = contexts.numpy().tolist()
    lengths = lengths.numpy().tolist()
    
    source_masks = get_masks(sources, pad_id)
    context_masks = get_masks(contexts, pad_id)
   
    features = {
        "source": torch.tensor(sources).long().cuda(),
        "context": torch.tensor(contexts).long().cuda(),
        "source_mask": torch.tensor(source_masks).float().cuda(),
        "context_mask": torch.tensor(context_masks).float().cuda(),
    }

    return features


class LM4MTDataIterator(object):
    """
    An iterator for loading batches of data into the transformer model.

    For training:

        Each batch contains tokens_in_batch target language tokens (approximately),
        target language sequences of the same length to minimize padding and therefore memory usage,
        source language sequences of very similar (if not the same) lengths to minimize padding and therefore memory usage.
        Batches are also shuffled.

    For validation and testing:

        Each batch contains just a single source-target pair, in the same order as in the files from which they were read.
    """

    def __init__(self, dataset, split, batch_size, pad_id=0):
        """
        :param data_folder: folder containing the source and target language data files
        :param source_suffix: the filename suffix for the source language files
        :param target_suffix: the filename suffix for the target language files
        :param split: train, or val, or test?
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.batch_size = batch_size
        self.pad_id = pad_id if pad_id is not None else -1
        # self.source_suffix = source_suffix
        # self.target_suffix = target_suffix
        assert split.lower() in {"train", "val",
                                 "test"}, "'split' must be one of 'train', 'val', 'test'! (case-insensitive)"
        self.split = split.lower()

        # Is this for training?
        self.for_training = self.split == "train"

       
        self.dataset = dataset.all_data # list(zip(source_data, target_data, source_lengths, target_lengths))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.for_training:
            self.dataset.sort(key=lambda x: x["input_len"])
                
        # Create batches
        self.create_batches()

    def __len__(self):
        return self.n_batches


    def create_batches(self):
        """
        Prepares batches for one epoch.
        """
        # If training
        if self.for_training:
            # Group or chunk based on target sequence lengths
            chunks = [list(g) for _, g in groupby(self.dataset, key=lambda x: x["input_len"])]

            # Create batches, each with the same target sequence length
            self.all_batches = list()
            for chunk in chunks:
                # Sort inside chunk by source sequence lengths, so that a batch would also have similar source sequence lengths
                chunk.sort(key=lambda x: x["input_len"])
                # How many sequences in each batch? Divide expected batch size (i.e. tokens) by target sequence length in this chunk
                seqs_per_batch = self.batch_size // chunk[0]["input_len"]
                # each batch must have at least 1 sequence
                seqs_per_batch = 1 if seqs_per_batch < 1 else seqs_per_batch 
                # Split chunk into batches
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            # Shuffle batches
            shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else:
            # Simply return once pair at a time
            line_idx = [(i, inst["input_len"]) for i, inst in enumerate(self.dataset)]
            sorted_input_lens = sorted(line_idx, key=lambda x: x[1], reverse=True)
            sorted_keys = {}
            sorted_inputs = []
            for i, (idx, _) in enumerate(sorted_input_lens):
                sorted_inputs.append(self.dataset[idx])
                sorted_keys[idx] = i
            self.sorted_keys = sorted_keys
            self.sorted_inputs = sorted_inputs
            self.all_batches = [self.sorted_inputs[i: i + self.batch_size] for i in range(0, len(self.sorted_inputs), self.batch_size)]
            # self.all_batches = [[d] for d in self.dataset]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
    
    def __iter__(self):
        """
        Iterators require this method defined.
        
        """
        self.current_batch += 1
        try:
            # source_data, target_data, source_lengths, target_lengths = zip(*self.all_batches[self.current_batch])
            source_data, target_data, ctx_data, input_max_len = [], [], [], []
            
            for inst in self.all_batches[self.current_batch]:
              ctx_data.append(inst['ctx'])
              source_data.append(inst['src'])
              input_max_len.append(inst['input_len'])
              if self.for_training:
                  target_data.append(inst['tgt'])
        # Stop iteration once all batches are iterated through
        except IndexError:
            raise StopIteration

        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data],
                                   batch_first=True,
                                   padding_value=self.pad_id)
        
        ctx_data = pad_sequence(sequences=[torch.LongTensor(c) for c in ctx_data],
                                batch_first=True,
                                padding_value=self.pad_id)
        source_mask = (source_data != self.pad_id).long()
        ctx_mask = (ctx_data != self.pad_id).long()
        
        if self.for_training:
            target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data],
                                   batch_first=True,
                                   padding_value=self.pad_id)
            
            target_mask = (target_data != self.pad_id).long()
            
            labels = target_data[:, 1:].contiguous()
            return {"source": torch.LongTensor(source_data),
                    "target": torch.LongTensor(target_data[:, :-1]),
                    "context": torch.LongTensor(ctx_data),
                    "source_mask": torch.LongTensor(source_mask),
                    "target_mask": torch.LongTensor(target_mask[:, :-1]),
                    "context_mask": torch.LongTensor(ctx_mask)}, torch.LongTensor(labels)
        else:
            return {"source": torch.LongTensor(source_data),
                    "context": torch.LongTensor(ctx_data),
                    "source_mask": torch.LongTensor(source_mask),
                    "context_mask": torch.LongTensor(ctx_mask)
            }
        
       

    def __next__(self):
        """
        Iterators require this method defined.

        :returns: the next batch, containing:
            source language sequences, a tensor of size (N, encoder_sequence_pad_length)
            target language sequences, a tensor of size (N, decoder_sequence_pad_length)
            true source language lengths, a tensor of size (N)
            true target language lengths, typically the same as decoder_sequence_pad_length as these sequences are bucketed by length, a tensor of size (N)
        """
        # Update current batch index
        return self
        
