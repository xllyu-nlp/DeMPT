# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
from modules.criterions import SmoothedCrossEntropyLoss
import thumt.modules as modules
import torch.distributed as dist
import thumt.utils.summary as summary


def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(*new_shape)
    return tensor.permute(0, 2, 1, 3)


def _combine_heads(x):
    batch = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    channels = x.shape[3]

    y = torch.transpose(x, 2, 1)

    return torch.reshape(y, [batch, length, heads * channels])

def _avg_pool(mask, x):
    # mask: [*, seq_len]
    # x: [*, seq_len, hidden]
    #[batch, 1, seq_len]
    # return [*, 1, hidden_size]
    mask_sum = mask.sum(dim=-1, keepdim=True).unsqueeze(-1) # [*, 1, 1]
    x_sum = (x * mask.unsqueeze(-1)).sum(dim=-2, keepdim=True) # [*, 1, hidden]
    x_avg = x_sum / mask_sum
    return x_avg

class Prompt(nn.Module):

    def __init__(self, model, num_prompts, prompt_length, name="prompt", prompt_proj=True):
        super(Prompt, self).__init__()
        self.embed_dim = model.config.hidden_size
        self.split_size = self.embed_dim
        self.hidden_size = model.config.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scales = nn.Parameter(
            torch.ones([num_prompts]))
        self.prompts_proj = prompt_proj
        self.register_parameter("scales", self.scales)
        # [num_prompts, 2 * num_decoder_layers, prompt_length, hidden_size]
        
        
        self.prompts = nn.Parameter(
            torch.empty(
            [
                num_prompts, 2 * self.num_decoder_layers,
                prompt_length, self.hidden_size
            ]))
        if self.prompts_proj:
           self.k_proj = nn.Sequential(
               nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
               nn.Tanh(),
               nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
           )
           self.v_proj = nn.Sequential(
               nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
               nn.Tanh(),
               nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
           )
        self.register_parameter("prompts", self.prompts)
        # initlize
        
        # self._model = [model]

        with torch.no_grad():
            for i in range(self.prompts.shape[0]):
                for j in range(self.prompts.shape[1]):
                    nn.init.xavier_uniform_(self.prompts[i, j])
            
            

    # @property
    # def model(self):
    #     return self._model[0]

    def forward(self, batch_size):
        # [num_promts]
        key_values = [[] for _ in range(self.prompts.shape[0])]

        for i in range(self.prompts.shape[0]):
            for j in range(self.num_decoder_layers):
                scale = torch.maximum(torch.ones([]), self.scales[i])        
                # [1, promt_length, hidden_size]
                k = self.prompts[i, 2*j][None, :, :] * scale
                v = self.prompts[i, 2*j+1][None, :, :] * scale
                # [batch_size, prompt_length, hidden_size]
                k = k.repeat([batch_size, 1, 1])
                v = v.repeat([batch_size, 1, 1])
                if self.prompts_proj:
                    k = self.k_proj(k)
                    v = self.v_proj(v)
                # [batch_size, num_heads, prompt_length, head_dim]
                k = _split_heads(k, self.num_heads, self.head_dim) 
                v = _split_heads(v, self.num_heads, self.head_dim)
                key_values[i].append((k, v))
        # pair of key_values = [num_promts, batch_size, num_heads, prompt_length, head_dim]
        # list: [num_prompts, ]
        
        return key_values


class OLM4MTModel(nn.Module):

    def __init__(self, model, params, name="lm4mt", is_train=True):

        # model is the model of PLM
        super(OLM4MTModel, self).__init__()
        self.params = params
        # Do not add plm parameters to our module
        self._lm_model = model
        # self.add_module("PretrainedLLM", self._lm_model)
        params.hidden_size = model.config.hidden_size
        self.is_train = is_train
        self.hidden_size = params.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if params.share_prompt:
            self.prompt_model = Prompt(model, 1, params.prompt_length)
        else:
            self.prompt_model = Prompt(model, 2+params.re_encoding,
                                       params.prompt_length)
        
        # initialize the key_types_embed
        if self.params.keys_type_num > 0:
            # [key_types_num, hidden_size]
            self.keys_type_embeddings = nn.Parameter(
                torch.empty(
                [
                    self.num_heads, params.keys_type_num, self.head_dim
                ]))
            
            self.register_parameter("key_type_embeddings", self.keys_type_embeddings)
            with torch.no_grad():    
                nn.init.xavier_uniform_(self.keys_type_embeddings)
            self.keys_type_dict = {"ctx_prompt":0, "source_prompt":1, "ctx_seq":2, "decode_prompt":3, "source_seq":4}
        
        self.criterion = SmoothedCrossEntropyLoss(
            params.label_smoothing)
        
        

    @property
    def lm_model(self):
        return self._lm_model
        # return self._lm_model[0]

    @property
    def src_embedding(self):
        return self.lm_model.get_input_embeddings().weight

    @property
    def tgt_embedding(self):
        return self.lm_model.get_input_embeddings().weight

    @property
    def softmax_embedding(self):
        return self.tgt_embedding

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"], strict=False)

    def generate_prefix(self):
        key_values = self.prompt_model.forward(1)

        state = {}

        for i, (k, v) in enumerate(key_values[0]):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["enc_key_%d" % i] = k
            state["enc_value_%d" % i] = v

        for i, (k, v) in enumerate(key_values[1]):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["rec_key_%d" % i] = k
            state["rec_value_%d" % i] = v

        for i, (k, v) in enumerate(key_values[2]):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["dec_key_%d" % i] = k
            state["dec_value_%d" % i] = v

        return state

    
    def encode(self, features, state, mode="infer"):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        key_values = self.prompt_model.forward(batch_size)
        past_key_values = key_values[0]
        
        for i in range(self.num_decoder_layers):
            # Prompt for the decoding stage
            key, value = key_values[-1][i]
            key = key + prompt_type_embedding
            pk, pv = pkv[i]
            pk = pk + src_type_embedding
            # Concat decoding prompt and re-encoded activations
            past_key_values.append((torch.cat([key, pk], axis=2),
                                    torch.cat([value, pv], axis=2)))
        # pair of (k, v): [batch, num_heads, seq_len+prompt_len, per_head_dim]
        state["past_key_values"] = past_key_values

        return state


    def decode(self, features, state, mode="infer", labels=None):
        
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        
        # src_mask = features["source_mask"]
        tgt_mask = features["source_mask"]
        
        state["inputs"] = input_ids

        pfx_mask = torch.ones([batch_size, self.params.prompt_length],
                              device=tgt_mask.device)
        attention_mask = torch.cat([pfx_mask, tgt_mask],
                                   dim=1)
        print("mask:", attention_mask.shape)
        if mode == "infer":

            input_ids = input_ids[:, -1:]
            # position_ids = position_ids[:, -1:]
        print("inputs_ids:", input_ids.shape)
        import pdb
        pdb.set_trace()
        if self.is_train:
            outputs = self.lm_model(input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    # position_ids=position_ids,
                                    use_cache=True, pooling_state=state, labels=labels)
        else:
            with torch.no_grad():
                outputs = self.lm_model(input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    # position_ids=position_ids,
                                    use_cache=True, pooling_state=state)
        # pooling past_k_v + outputs?
        state["past_key_values"] = outputs.past_key_values
        # logits = outputs.logits
        if mode == "infer":
            logits = outputs.logits
            logits = logits[:, 0, :]
            return logits, state
        else:
            return outputs, state
        

    def forward(self, features, labels):
        # mask = features["target_mask"]
        
        state = {}
        
        state = self.encode(features, state)
        
        outputs, state = self.decode(features, state, "train", labels=labels)
        self.state = state

        return outputs.loss

    def empty_state(self, batch_size, device):
        return {}

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            prompt_length=128,
            label_smoothing=0.1,
            sep_id=2,
            dec_no_prefix=False,
            share_prompt=False,
            re_encoding=1
        )

        return params

    @staticmethod
    def default_params(name=None):
        return LM4MTModel.base_params()
