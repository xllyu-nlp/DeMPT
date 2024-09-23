
from transformers import LlamaModel, LlamaForCausalLM, LlamaConfig, BloomConfig, BloomForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class LlamaForMSP(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def mspecoding_forward():
        pass
    
    def mspencoding_forward():
        pass

class GateFusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, out_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.1)
    
    def forward(self, x1, x2):
        return self.dropout(self.sigmoid(self.linear(torch.cat((x1, x2), dim=-1))) * x2)

    
class LlamaCLForMSP(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
    
    def add_new_params(self, params):
       self.enhance_decode = params.enhance_decode
       self.enhance_proj = params.enhance_proj
       self.enhance_attn = params.enhance_attn
       
       if self.enhance_decode and self.enhance_attn:
           self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
           self.q_scaling_product_ctx_gate = GateFusion(self.config.hidden_size*2, self.config.hidden_size)
           self.q_scaling_product_src_gate = GateFusion(self.config.hidden_size*2, self.config.hidden_size)
           self.q_scaling_product_norm = torch.nn.LayerNorm(self.config.hidden_size)
           self.q_scaling_product_proj = nn.Sequential(
              nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
              nn.Tanh(),
              nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
           )
       
       if self.enhance_decode and self.enhance_proj:
        #   self.source_info_proj = nn.Sequential(
        #       nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
        #       nn.Tanh(),
        #       nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
        #   )
          
          # self.dropout = nn.Dropout(p=0.1)
          self.source_info_proj_src_dense = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size*2, out_features=self.config.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
          )
          
          self.source_info_proj_ctx_dense = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size*3, out_features=self.config.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
          )
       
    


    
    def naive_cross_attn(self, q, k, v, q_m, k_m):
        # q: [bs, tgt_len, hidden_size]
        # k, v: [bs, src_len, hidden_size]
        # q_m: [bs, tgt_len]
        # k_m: [bs, src_len]
        q_m_b = (1 - q_m).bool()
        k_m_b = (1 - k_m).bool()
        scaling = float(q.shape[-1]) ** -0.5
        query = self.q_scaling_product.mul_scalar(q, scaling)
        attn_w = torch.bmm(query, k.transpose(1, 2)) # [bs, tgt_len, src_len]
        # import pdb
        # pdb.set_trace()
        # attn_w.masked_fill_(q_m_b.unsqueeze(2), float('-inf')) # [bs, tgt_len, src_len]
        attn_w.masked_fill_(k_m_b.unsqueeze(1), float('-inf')) # [bs, tgt_len, src_len]
        attn_w = torch.nn.functional.softmax(attn_w, dim=-1)
        
        attn_output = torch.bmm(attn_w, v) # [bs, tgt_len, hidden_size]
        
        return self.q_scaling_product_norm(attn_output)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        mode: Optional[str] = "decoder", 
        pooling_state: Optional[dict] = None,
        logit_w: Optional[torch.Tensor] = 1.0
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        
        if mode == "decoder":
            if self.enhance_decode and pooling_state is not None:
                
                if self.enhance_attn:
                    src_enhance_hidden = self.naive_cross_attn(hidden_states, pooling_state['src_state'], pooling_state['src_state'],
                                                        pooling_state['tgt_mask'], pooling_state['src_mask'])
                    src_enhance_hidden = self.q_scaling_product_src_gate(hidden_states, src_enhance_hidden)
                    ctx_enhance_hidden = self.naive_cross_attn(hidden_states, pooling_state['ctx_state'], pooling_state['ctx_state'],
                                                        pooling_state['tgt_mask'], pooling_state['ctx_mask'])
                    ctx_enhance_hidden = self.q_scaling_product_ctx_gate(hidden_states, ctx_enhance_hidden)
                    # print("ctx:", ctx_enhance_hidden.shape)
                    # print("src:", src_enhance_hidden.shape)
                    # print("hid", hidden_states.shape)
                    # print("gogoooo")
                    # exit(0)
                    hidden_states = hidden_states + ctx_enhance_hidden + src_enhance_hidden
                    logits = self.lm_head(self.q_scaling_product_proj(hidden_states))
                    
                
                # size of elements in pooling_state: [batch, 1, hidden]
                if self.enhance_proj:
                    # pooling_state["src_pooling_state"] = self.dropout(self.source_info_proj(pooling_state["src_pooling_state"]))
                    # pooling_state["ctx_pooling_state"] = self.dropout(self.source_info_proj(pooling_state["ctx_pooling_state"]))
                    seq_len = hidden_states.shape[1]
                    rp_src = pooling_state["src_pooling_state"].repeat_interleave(repeats=seq_len,dim=1)
                    rp_ctx = pooling_state["ctx_pooling_state"].repeat_interleave(repeats=seq_len,dim=1)
                    src_dec_states = torch.cat((rp_src, hidden_states), dim=-1)
                    src_dec_logits = self.lm_head(self.source_info_proj_src_dense(src_dec_states))
                    # ctx_dec_states = pooling_state["ctx_pooling_state"] + hidden_states
                    # src_ctx_dec_state = pooling_state["ctx_pooling_state"] + pooling_state["src_pooling_state"] + hidden_states
                    src_ctx_dec_states = torch.cat((rp_ctx, rp_src, hidden_states), dim=-1)
                    src_ctx_dec_logits = self.lm_head(self.source_info_proj_ctx_dense(src_ctx_dec_states))
                    dec_logits = self.lm_head(hidden_states)
                    logits = (logit_w * src_ctx_dec_logits + logit_w * src_dec_logits + dec_logits)/3
                
            else:
                logits = self.lm_head(hidden_states)
        
        else:
            logits = 0.0
        
        outputs.hidden_states = hidden_states


        #logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous() 
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
       
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BloomCLForMSP(BloomForCausalLM):

    def __init__(self, config):
        super().__init__(config)

    def convert_to_standard_cache(self,
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length).transpose(2, 3).contiguous(),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, seq_length, head_dim = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].transpose(2, 3).contiguous().view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )  



    def add_new_params(self, params):
       self.enhance_decode = params.enhance_decode
       self.enhance_proj = params.enhance_proj
       self.enhance_attn = params.enhance_attn
       
       if self.enhance_decode and self.enhance_attn:
           self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
           self.q_scaling_product_ctx_gate = GateFusion(self.config.hidden_size*2, self.config.hidden_size)
           self.q_scaling_product_src_gate = GateFusion(self.config.hidden_size*2, self.config.hidden_size)
           self.q_scaling_product_norm = torch.nn.LayerNorm(self.config.hidden_size)
           self.q_scaling_product_proj = nn.Sequential(
              nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
              nn.Tanh(),
              nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
           )
       
       if self.enhance_decode and self.enhance_proj:
        
            self.source_info_proj_src_dense = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size*2, out_features=self.config.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
            )
            
            self.source_info_proj_ctx_dense = nn.Sequential(
                nn.Linear(in_features=self.config.hidden_size*3, out_features=self.config.hidden_size, bias=False),
                nn.Tanh(),
                nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size, bias=False),
            )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mode: Optional[str] = "decoder", 
        pooling_state:  Optional[dict] = None,
        logit_w: Optional[torch.Tensor] = 1.0,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if past_key_values is not None:
            past_key_values = self.convert_to_bloom_cache(past_key_value=past_key_values)
        
    
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        
        if mode == "decoder":
            if self.enhance_decode and pooling_state is not None:
                
                if self.enhance_attn:
                    src_enhance_hidden = self.naive_cross_attn(hidden_states, pooling_state['src_state'], pooling_state['src_state'],
                                                        pooling_state['tgt_mask'], pooling_state['src_mask'])
                    src_enhance_hidden = self.q_scaling_product_src_gate(hidden_states, src_enhance_hidden)
                    ctx_enhance_hidden = self.naive_cross_attn(hidden_states, pooling_state['ctx_state'], pooling_state['ctx_state'],
                                                        pooling_state['tgt_mask'], pooling_state['ctx_mask'])
                    ctx_enhance_hidden = self.q_scaling_product_ctx_gate(hidden_states, ctx_enhance_hidden)
                    # print("ctx:", ctx_enhance_hidden.shape)
                    # print("src:", src_enhance_hidden.shape)
                    # print("hid", hidden_states.shape)
                    # print("gogoooo")
                    # exit(0)
                    hidden_states = hidden_states + ctx_enhance_hidden + src_enhance_hidden
                    logits = self.lm_head(self.q_scaling_product_proj(hidden_states))
                    
                
                # size of elements in pooling_state: [batch, 1, hidden]
                if self.enhance_proj:
                    # pooling_state["src_pooling_state"] = self.dropout(self.source_info_proj(pooling_state["src_pooling_state"]))
                    # pooling_state["ctx_pooling_state"] = self.dropout(self.source_info_proj(pooling_state["ctx_pooling_state"]))
                    seq_len = hidden_states.shape[1]
                    rp_src = pooling_state["src_pooling_state"].repeat_interleave(repeats=seq_len,dim=1)
                    rp_ctx = pooling_state["ctx_pooling_state"].repeat_interleave(repeats=seq_len,dim=1)
                    src_dec_states = torch.cat((rp_src, hidden_states), dim=-1)
                    src_dec_logits = self.lm_head(self.source_info_proj_src_dense(src_dec_states))
                    # ctx_dec_states = pooling_state["ctx_pooling_state"] + hidden_states
                    # src_ctx_dec_state = pooling_state["ctx_pooling_state"] + pooling_state["src_pooling_state"] + hidden_states
                    src_ctx_dec_states = torch.cat((rp_ctx, rp_src, hidden_states), dim=-1)
                    src_ctx_dec_logits = self.lm_head(self.source_info_proj_ctx_dense(src_ctx_dec_states))
                    dec_logits = self.lm_head(hidden_states)
                    lm_logits = (logit_w * src_ctx_dec_logits + logit_w * src_dec_logits + dec_logits)/3
            else:
                lm_logits = self.lm_head(hidden_states)
        
        else:
            lm_logits = 0.0
        
        transformer_outputs.hidden_states = hidden_states
        
        # lm_logits = self.lm_head(hidden_states)

        
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits.contiguous()
            shift_labels = labels.contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
      
        batch_size = hidden_states.shape[0]
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=self.convert_to_standard_cache(past_key_value=transformer_outputs.past_key_values, batch_size=batch_size),
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
