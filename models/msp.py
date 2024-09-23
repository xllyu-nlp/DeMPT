# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils

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
class Prefix(nn.Module):
    
    def __init__(self, model, prompt_length, hidden_size):
        super(Prefix, self).__init__()
        self._num_hidden_layers = model.config.num_hidden_layers
        self._prompt_length = prompt_length
        self._hidden_size = hidden_size
        self._emb_size = model.config.hidden_size
        self._num_heads = model.config.num_attention_heads
        self._head_dim = self._emb_size // self._num_heads
        
        self.emb = nn.Parameter(
            torch.empty([prompt_length, self._emb_size]))
        
        self.mlp1 = nn.Linear(self._emb_size, hidden_size,
                                    bias=False)
        self.mlp2 = nn.Linear(hidden_size,
            self._emb_size * 2 * self._num_hidden_layers,
            bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)

    def forward(self, batch_size):
        prefix_cat = self.mlp2(torch.tanh(self.mlp1(self.emb)))
        prefix_list = torch.reshape(
            prefix_cat,
            [-1, self._emb_size, 2 * self._num_hidden_layers])

        prefix_list = torch.unbind(prefix_list, -1)

        prefixes = [[]]

        for i in range(self._num_hidden_layers):
            k = prefix_list[2*i]
            v = prefix_list[2*i + 1]
            k = k.unsqueeze(0).repeat([batch_size, 1, 1])
            v = v.unsqueeze(0).repeat([batch_size, 1, 1])
            k = _split_heads(k, self._num_heads, self._head_dim)
            v = _split_heads(v, self._num_heads, self._head_dim)
            prefixes[0].append((k.contiguous(), v.contiguous()))

        return prefixes


class Prompt(nn.Module):

    def __init__(self, model, num_prompts, prompt_length, name="prompt", prompts_proj=True, share_proj_layer=True):
        super(Prompt, self).__init__()
        self.embed_dim = model.config.hidden_size
        self.split_size = self.embed_dim
        self.hidden_size = model.config.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scales = nn.Parameter(
            torch.ones([num_prompts]))
        self.prompts_proj = prompts_proj
        self.share_proj_layer = share_proj_layer
        self.register_parameter("scales", self.scales)
        # [num_prompts, 2 * num_decoder_layers, prompt_length, hidden_size]
        
        self.prompts = nn.Parameter(
            torch.empty(
            [
                num_prompts, 2 * self.num_decoder_layers,
                prompt_length, self.hidden_size
            ]))
        if self.prompts_proj:
            if self.share_proj_layer:
                self.k_proj = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
                )
                self.v_proj = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
                )
            else:
                self.k_proj_in = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
                self.v_proj_in = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False),
                self.k_proj_list =  nn.ModuleList([nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
                ) for _ in range(self.num_decoder_layers)])
                self.v_proj_list =  nn.ModuleList([nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
                ) for _ in range(self.num_decoder_layers)])
           # self.dropout = nn.Dropout(p=0.2)
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
                    if self.share_proj_layer:
                        k = self.k_proj(k)
                        v = self.v_proj(v)
                    else:
                        k = self.k_proj_list[j](k)
                        v = self.k_proj_list[j](v)
                # [batch_size, num_heads, prompt_length, head_dim]
                k = _split_heads(k, self.num_heads, self.head_dim) 
                v = _split_heads(v, self.num_heads, self.head_dim)
                key_values[i].append((k.contiguous(), v.contiguous()))
    
        return key_values


# New feature: added by Xinglin
class LM4MTModel(nn.Module):

    def __init__(self, model, params, name="lm4mt", is_train=True):

        # model is the model of PLM
        super(LM4MTModel, self).__init__()
        self.params = params
        self.soruce_annealing = params.source_annealing
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
        self.dec_no_prefix = self.params.re_encoding == 0
        self.convert_dec_mask = self.params.re_encoding == 0 and self.params.use_ctx
        
        if self.dec_no_prefix:
            self.prompt_model = Prefix(model, params.prompt_length, self.embed_dim)
        else:
            if params.share_prompt:
                self.prompt_model = Prompt(model, 1, params.prompt_length, prompts_proj=params.prompts_proj, share_proj_layer=params.share_proj_layer)
                
            else:
                self.prompt_model = Prompt(model, 2+params.re_encoding,
                                        params.prompt_length, prompts_proj=params.prompts_proj, share_proj_layer=params.share_proj_layer)
               
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
        
        self.load_state_dict(state["model"], strict=True)

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

    def encode(self, features, state):
        # [batch_size, seq_len]
        if self.convert_dec_mask:
            features["source"] = features["context"]
            features["source_mask"] = features["context_mask"]
        
        if self.dec_no_prefix and not self.convert_dec_mask:
            features["context"] = features["source"]
            features["context_mask"] = features["source_mask"]
        
        input_ids = features["context"]
        batch_size = input_ids.shape[0]
        pl = self.params.prompt_length
        
        key_values = self.prompt_model.forward(batch_size)
        
        # adding the key type
        if self.params.keys_type_num > 0:
            past_key_values = []
            # [1, num_head, 1, head_dim]
            type_embedding = self.keys_type_embeddings[:, self.keys_type_dict["ctx_prompt"], :].unsqueeze(0).unsqueeze(2)
            for (k, v) in key_values[0]:
               past_key_values.append((k + type_embedding, v)) 
            
        # Prompt for the encoding stage
        else:
            past_key_values = key_values[0]
        
        if self.is_train:
            outputs = self.lm_model(input_ids, past_key_values=past_key_values, 
                                use_cache=True, mode="encoder")
            
        else:
            with torch.no_grad():
                outputs = self.lm_model(input_ids, past_key_values=past_key_values,
                                use_cache=True, mode="encoder")

        past_key_values = []
        state["ctx_state"] = outputs.hidden_states
        state["ctx_mask"] = features["context_mask"]
        for (k, v) in outputs.past_key_values:
            # Activations in the encoding stage
            if self.dec_no_prefix:
                past_key_values.append((k, v))
            else:
                past_key_values.append((k[:, :, pl:, :], v[:, :, pl:, :]))
        
        # enhance the utilization of src and ctx information when decoding
        if self.params.enhance_decode:
           pooling_mask = features["context_mask"] #[batch, seq_len]
           pooling_ctx = _avg_pool(pooling_mask, outputs.hidden_states)
           state["ctx_pooling_state"] = pooling_ctx
        
        # pair of (k, v): [batch, num_heads, seq_len, per_head_dim]  
        state["enc_activations"] = tuple(past_key_values)
        # pair of (k, v): [batch, num_heads, prompt_len, per_head_dim]  
        state["key_values"] = key_values
        # pair of (k, v): [batch, num_heads, seq_len, per_head_dim]  
        state["past_key_values"] = tuple(past_key_values)
        
        
       
        
        # Re-encoding
        for i in range(self.params.re_encoding):
            state = self.rencode(features, state, i)

        # Prepare for decoding
        pkv = state["past_key_values"]
        past_key_values = []
        if self.params.keys_type_num > 0:
            prompt_type_embedding = self.keys_type_embeddings[:, self.keys_type_dict["decode_prompt"], :].unsqueeze(0).unsqueeze(2)
            src_type_embedding = self.keys_type_embeddings[:, self.keys_type_dict["source_seq"], :].unsqueeze(0).unsqueeze(2)
        else:
            prompt_type_embedding = 0.0
            src_type_embedding = 0.0 
        
        if self.dec_no_prefix:
            return state
        
        else:
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

    def rencode(self, features, state, idx):
        # Re-encoding

        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        pl = self.params.prompt_length
        sl = features["context"].shape[1]
        src_mask = features["source_mask"]
        context_mask = features["context_mask"]
        # 3 x 2
        key_values = state["key_values"]
        src_length = torch.sum(features["source_mask"], 1).long()

        pkv = state["past_key_values"]

        pfx_mask = torch.ones([batch_size, pl], device=src_mask.device)
        attention_mask = torch.cat([pfx_mask, context_mask, src_mask], dim=1)
        
        # Add prefixes
        past_key_values = []
        if self.params.keys_type_num > 0:
            prompt_type_embedding = self.keys_type_embeddings[:, self.keys_type_dict["source_prompt"], :].unsqueeze(0).unsqueeze(2)
            ctx_type_embedding = self.keys_type_embeddings[:, self.keys_type_dict["ctx_seq"], :].unsqueeze(0).unsqueeze(2)
        else:
            prompt_type_embedding = 0.0
            ctx_type_embedding = 0.0 

        for i in range(self.num_decoder_layers):
            if not self.params.share_prompt:
                # Prompt for the re-encoding stage
                # [batch, num_heads, prompt_len, per_head_dim]
                key, value = key_values[1+idx][i]
                key = prompt_type_embedding + key
                
            else:
                key, value = key_values[0][i]
                key = prompt_type_embedding + key
            # [batch, num_heads, seq_len, per_head_dim]
            pk, pv = pkv[i]
            pk = ctx_type_embedding + pk
            # [batch, num_heads, seq_len+prompt_len, per_head_dim]
            past_key_values.append((torch.cat([key, pk], axis=2),
                                    torch.cat([value, pv], axis=2)))
            
        if self.is_train:
            outputs = self.lm_model(input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    use_cache=True, mode="encoder")
        else:
            with torch.no_grad():
                outputs = self.lm_model(input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    use_cache=True, mode="encoder")
        
    
        
        past_key_values = []
        state["src_state"] = outputs.hidden_states
        state["src_mask"] = src_mask
        for (k, v) in outputs.past_key_values:
            # Activations in the re-encoding stage
            # [batch, num_heads, seq_len, per_head_dim]
            past_key_values.append((k[:, :, pl+sl:, :], v[:, :, pl+sl:, :]))
        
        # enchance utilization of the src information 
        if self.params.enhance_decode:
           pooling_mask = src_mask #[batch, 1, seq_len]
           pooling_src = _avg_pool(pooling_mask, outputs.hidden_states) # [batch, 1, hidden_size]
           state["src_pooling_state"] = pooling_src

        state["past_key_values"] = tuple(past_key_values)

        return state

    def decode(self, features, state, mode="infer", labels=None, time=None):
        input_ids = features["target"]
        batch_size = input_ids.shape[0]
        src_mask = features["source_mask"]
        tgt_mask = features["target_mask"]
        state["target"] = input_ids
        state["tgt_mask"] = tgt_mask
        
        if self.params.enhance_decode and self.soruce_annealing:
            if mode == "infer" and time is not None:
                logit_w = time / (src_mask.sum(-1, keepdim=True) * 1.2) # [batch_size * beam, 1]
                logit_w = logit_w.unsqueeze(2) # [batch_size * beam, 1, 1]
            else:
                total_step = src_mask.sum(-1, keepdim=True) * 1.2 # [batch_size, 1]
                step_i = torch.range(start=1, end=tgt_mask.shape[1], device=tgt_mask.device).unsqueeze(0) # [1, tgt_len]
                step_i = torch.repeat_interleave(step_i, repeats=tgt_mask.shape[0], dim=0) # [batch_size, tgt_len]
                logit_w = (step_i / total_step).unsqueeze(2) # [batch_size, tgt_len, 1]
                
        else:
            logit_w = 1.0
            
        pfx_mask = torch.ones([batch_size, self.params.prompt_length],
                            device=src_mask.device)
        attention_mask = torch.cat([pfx_mask, src_mask, tgt_mask],
                                dim=1)
        past_key_values = state["past_key_values"]

        if mode == "infer":
            input_ids = input_ids[:, -1:]
           
        if self.is_train:
            outputs = self.lm_model(input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    use_cache=True, pooling_state=state, labels=labels, logit_w=logit_w)
        else:
            with torch.no_grad():
                outputs = self.lm_model(input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    use_cache=True, pooling_state=state)
        state["past_key_values"] = outputs.past_key_values
        
        if mode == "infer":
            logits = outputs.logits
            logits = logits[:, 0, :]
            return logits, state
        else:
            return outputs, state

    def forward(self, features, labels):
        mask = features["target_mask"]
        state = {}
        
        state = self.encode(features, state)
        outputs, state = self.decode(features, state, "train", labels)
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
