from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM, get_peft_config, LoraModel
from transformers import AutoTokenizer, LlamaTokenizer, LlamaConfig
from transformers.generation.utils import * 
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, ModuleList, Linear
import json
import os
from copy import deepcopy


class MHLlamaConfig(LlamaConfig):
    model_type = "mhllama"
    
    def __init__(
            self,
            num_heads=5,
            head_assembly='follow',
            head_loss='rearrange',
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            col_token_id=3,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
        **kwargs,):
        super().__init__(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
        **kwargs,)
        self.num_heads = num_heads
        self.head_assembly = head_assembly
        self.head_loss = head_loss
        self.col_token_id = col_token_id


class MOELlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MHLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = ModuleList(config.num_heads*[self.mlp])
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        column: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp[column](hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MOELlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MOELlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        column: int,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    column,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    column,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if dtype == torch.bfloat16:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=torch.float, device=device)
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class MultiheadLlamaForCausalLM(LlamaPreTrainedModel):
    config_class = MHLlamaConfig
    
    def __init__(self, config):
        super().__init__(config) #['llamamodel_config']
        self.model = MOELlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_heads
        self.mh_config = config
        
        headlist = []
        # self.heads = nn.ModuleList()
        for i in range(config.num_heads):
            # self.heads.add_module(f'lm_head_{i}', nn.Linear(config.hidden_size, config.vocab_size, bias=False))
            headlist.append(  nn.Linear(config.hidden_size, config.vocab_size, bias=False)) 
        self.heads = nn.ModuleList(headlist)
        
        self.post_init() # Initialize weights and apply final processing
        self.trace = False # for debugging
        

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.heads

    def set_output_embeddings(self, new_embeddings):
        self.heads = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    
    def get_heads_logits(self, hidden_states, input_ids, insert_indices, return_individual_preds=False):
        """return_individual_preds is if you want to see what is output by each individual head"""
        print('in get_heads_logits, hidden states', hidden_states.shape)
        preds = []
        for i in range(self.num_heads):
            # print('\t', i, insert_indices[i]+1, insert_indices[i+1])
            pred = self.heads[i](hidden_states.float()) # batch_size x tokens x vocab_size
            preds.extend((pred[:, insert_indices[i]+1:insert_indices[i+1], :], pred[:, -1:, :]))
        preds.append(pred[:, insert_indices[-1]+1:, :]) # just to get the last col_token marker after the last head, plus a filler token at very end
        # print('input ids shape', input_ids.shape)
        # print('shapes in preds', [p.shape for p in preds])
        # print('assembled preds', torch.cat(preds, dim=1).shape)
        return torch.cat(preds, dim=1) # batch_size x tokens x vocab_size
    
    
    def insert_cols_to_prompt(self, prompt, cols):
        """prompt: 1 x tokensp tensor
            cols: batch_size x num_cols x tokensc tensor
            returns sents: a batch_size x tokensp+num_cols*tokensc tensor, with cols inserted at cls_token locations"""
        batch_size = cols.shape[0]
        indices = torch.where(prompt==self.mh_config.col_token_id)[0] + 1 #so col_token_id will be at end of each split, not the start
        # tuple with num_cols elements. i-th element is prompt between (i-1)th and ith head, plus i-th col token:
        splits = torch.tensor_split(prompt, indices.cpu()) 
        splitscols = []
        for i, split in enumerate(splits):
            splitm = split.repeat(batch_size, 1) #batch_size x tokens
            splitcol = torch.cat((splitm[:, :-1], cols[:, i, :], splitm[:, -1]), dim=1) #dim1 changes, dim0 stay same
            #                       prompt chunch   col target/pred     col token
            splitscols.append(splitcol)
        return torch.cat(splitscols, dim=1)
        
    
    def forward(
        self,
        head_inds: torch.Tensor,
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
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Modified from LlamaForCausalLM forward()
        ```"""
        if self.trace: 
            print('in forward')
            print(input_ids, attention_mask, labels)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # blanks = torch.ones_like(labels) # batch_size x num_cols x targets_len tensor
        # inputs = self.insert_cols_to_prompt(input_ids, blanks) #batch_size x total_tokens

        if self.trace:
            print('in forward. input_ids', input_ids, 'attention mask', attention_mask, 'label', labels)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if len(head_inds.shape) == 0: # for generation, where we go token by token thus column is known
            col = head_inds.item()
            outputs = self.model(
                column=col,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0] # batch_size x tokens x 4096
            # print(hidden_states.shape)

            if self.config.pretraining_tp > 1:
                return ValueError('unsupported hyperparameter pretraining tp > 1')
            else:
                logits = self.heads[col](hidden_states)
                
            logits = logits.float() # logits are batch_size x tokens x 32000 (vocab size)

            loss = None
            if labels is not None:
                # print('logits', logits.shape, 'labels', labels.shape)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # print('logits', shift_logits.shape, 'labels', shift_labels.shape)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # print('logits', shift_logits.shape, 'labels', shift_labels.shape)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, #hidden_states, #
                attentions=outputs.attentions,
            )
        else: # for training, where column isn't specified
            outputs = []
            hidden_states = []
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.vocab_size)) #batchsize x tokens x vocab
            for i in range(self.num_heads):
                output = self.model(
                    column=i,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
                outputs.append(output)

                hidden_states = output[0] # batch_size x tokens x 4096
            # print(hidden_states.shape)

                if self.config.pretraining_tp > 1:
                    return ValueError('unsupported hyperparameter pretraining tp > 1')
                else:
                    logit = self.heads[i](hidden_states)
                    logit = logit.float() # logits are batch_size x tokens x 32000 (vocab size)
                    logits=logits.to(logit.device)
                    logits[:, torch.where(head_inds == i)[0], :] = logit[:, torch.where(head_inds == i)[0], :]

            loss = None
            if labels is not None:
                # print('logits', logits.shape, 'labels', labels.shape)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous() # TODO? is this needed still?
                shift_labels = labels[..., 1:].contiguous()
                # print('logits', shift_logits.shape, 'labels', shift_labels.shape)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                # print('logits', shift_logits.shape, 'labels', shift_labels.shape)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) #+ outputs[1:]
                return (loss,) #+ output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                # past_key_values=outputs.past_key_values,
                # hidden_states=outputs.hidden_states, #hidden_states, #
                # attentions=outputs.attentions,
            )
        
        
    def can_generate(self): return True
    
    def set_trace(self, value):
        self.trace = value
    
    
    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        head_inds: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        
        # cur_tok = cur_len
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # forward pass to get next token
            outputs = self(
                head_inds=head_inds[cur_len],
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if eos_token was found in one sentence, set sentence to finished
            
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        
    
    
    
    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: torch.Tensor,
    #     insert_indices: torch.LongTensor,
    #     max_tokens_per_col: int,
    #     generation_config: Optional[GenerationConfig] = None,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #     synced_gpus: Optional[bool] = None,
    #     assistant_model: Optional["PreTrainedModel"] = None,
    #     streamer: Optional["BaseStreamer"] = None,
    #     negative_prompt_ids: Optional[torch.Tensor] = None,
    #     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    #     **kwargs,
    # ):
    #     r"""
    #     Copied from GenerationMixin (super) generate()
    #     """
    #     if self.trace: print('in generate')

    #     if synced_gpus is None:
    #         if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
    #             synced_gpus = True
    #         else:
    #             synced_gpus = False

    #     # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    #     self._validate_model_class()

    #     # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    #     if generation_config is None:
    #         # legacy: users may modify the model configuration to control generation -- update the generation config
    #         # model attribute accordingly, if it was created from the model config
    #         if self.generation_config._from_model_config:
    #             new_generation_config = GenerationConfig.from_model_config(self.config)
    #             if new_generation_config != self.generation_config:
    #                 warnings.warn(
    #                     "You have modified the pretrained model configuration to control generation. This is a"
    #                     " deprecated strategy to control generation and will be removed soon, in a future version."
    #                     " Please use a generation configuration file (see"
    #                     " https://huggingface.co/docs/transformers/main_classes/text_generation )"
    #                 )
    #                 self.generation_config = new_generation_config
    #         generation_config = self.generation_config

    #     generation_config = deepcopy(generation_config)
    #     model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    #     generation_config.validate()
    #     self._validate_model_kwargs(model_kwargs.copy())

    #     # 2. Set generation parameters if not already defined
    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    #     if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
    #         if model_kwargs.get("attention_mask", None) is None:
    #             logger.warning(
    #                 "The attention mask and the pad token id were not set. As a consequence, you may observe "
    #                 "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
    #             )
    #         eos_token_id = generation_config.eos_token_id
    #         if isinstance(eos_token_id, list):
    #             eos_token_id = eos_token_id[0]
    #         logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
    #         generation_config.pad_token_id = eos_token_id

    #     # 3. Define model inputs
    #     # inputs_tensor has to be defined
    #     # model_input_name is defined if model-specific keyword input is passed
    #     # otherwise model_input_name is None
    #     # all model-specific keyword inputs are removed from `model_kwargs`
    #     inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
    #         inputs, generation_config.bos_token_id, model_kwargs
    #     )
    #     batch_size = inputs_tensor.shape[0]

    #     # 4. Define other model kwargs
    #     model_kwargs["output_attentions"] = generation_config.output_attentions
    #     model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    #     # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    #     # generating the first new token or not, and we only want to use the embeddings for the first new token)
    #     if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
    #         model_kwargs["use_cache"] = True
    #     else:
    #         model_kwargs["use_cache"] = generation_config.use_cache

    #     accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    #     requires_attention_mask = "encoder_outputs" not in model_kwargs

    #     if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
    #         model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
    #             inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
    #         )

    #     # decoder-only models should use left-padding for generation
    #     if not self.config.is_encoder_decoder:
    #         # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
    #         # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
    #         if (
    #             generation_config.pad_token_id is not None
    #             and len(inputs_tensor.shape) == 2
    #             and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
    #         ):
    #             logger.warning(
    #                 "A decoder-only architecture is being used, but right-padding was detected! For correct "
    #                 "generation results, please set `padding_side='left'` when initializing the tokenizer."
    #             )

    #     if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
    #         # if model is encoder decoder encoder_outputs are created
    #         # and added to `model_kwargs`
    #         model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
    #             inputs_tensor, model_kwargs, model_input_name
    #         )

    #     # 5. Prepare `input_ids` which will be used for auto-regressive generation
    #     if self.config.is_encoder_decoder:
    #         input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
    #             batch_size=batch_size,
    #             model_input_name=model_input_name,
    #             model_kwargs=model_kwargs,
    #             decoder_start_token_id=generation_config.decoder_start_token_id,
    #             bos_token_id=generation_config.bos_token_id,
    #             device=inputs_tensor.device,
    #         )
    #     else:
    #         input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    #     # if streamer is not None:
    #     #     streamer.put(input_ids.cpu())
    #     input_ids.to(self.device)

    #     # 6. Prepare `max_length` depending on other stopping criteria.
    #     input_ids_seq_length = input_ids.shape[-1]
    #     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    #     if has_default_max_length and generation_config.max_new_tokens is None:
    #         warnings.warn(
    #             f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
    #             "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
    #             " recommend using `max_new_tokens` to control the maximum length of the generation.",
    #             UserWarning,
    #         )
    #     elif generation_config.max_new_tokens is not None:
    #         if not has_default_max_length:
    #             logger.warning(
    #                 f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
    #                 f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
    #                 "Please refer to the documentation for more information. "
    #                 "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
    #             )
    #         generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

    #     if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
    #         raise ValueError(
    #             f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
    #             f" the maximum length ({generation_config.max_length})"
    #         )
    #     if input_ids_seq_length >= generation_config.max_length:
    #         input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    #         logger.warning(
    #             f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
    #             f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
    #             " increasing `max_new_tokens`."
    #         )

    #     # 7. determine generation mode

    #     if self.device.type != input_ids.device.type:
    #         warnings.warn(
    #             "You are calling .generate() with the `input_ids` being on a device type different"
    #             f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
    #             f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
    #             " Please make sure that you have put `input_ids` to the"
    #             f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
    #             " running `.generate()`.",
    #             UserWarning,
    #         )

    #     # 8. prepare distribution pre_processing samplers
    #     logits_processor = self._get_logits_processor(
    #         generation_config=generation_config,
    #         input_ids_seq_length=input_ids_seq_length,
    #         encoder_input_ids=inputs_tensor,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         logits_processor=logits_processor,
    #     )

    #     # 9. prepare stopping criteria
    #     stopping_criteria = self._get_stopping_criteria(
    #         generation_config=generation_config, stopping_criteria=stopping_criteria
    #     )
    #     # 10. go into different generation modes
    #     # 11. run mlm sample
    #     return self.mlm_sample(
    #         input_ids,
    #         insert_indices,
    #         max_tokens_per_col,
    #         logits_processor=logits_processor,
    #         # stopping_criteria=stopping_criteria,
    #         pad_token_id=generation_config.pad_token_id,
    #         eos_token_id=generation_config.eos_token_id,
    #         output_scores=generation_config.output_scores,
    #         return_dict_in_generate=generation_config.return_dict_in_generate,
    #         synced_gpus=synced_gpus,
    #         streamer=streamer,
    #         **model_kwargs,
    #     )

    
    # def mlm_sample(
    #     self,
    #     input_ids: torch.LongTensor,
    #     insert_indices: torch.LongTensor,
    #     max_tokens_per_col: int,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     # stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     # max_length: Optional[int] = None,
    #     pad_token_id: Optional[int] = None,
    #     eos_token_id: Optional[Union[int, List[int]]] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_scores: Optional[bool] = None,
    #     output_logits: Optional[bool] = None,
    #     return_dict_in_generate: Optional[bool] = None,
    #     synced_gpus: bool = False,
    #     streamer: Optional["BaseStreamer"] = None,
    #     **model_kwargs,
    # ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Copied/modified from generation mixin greedy search

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        ```"""
        print('in mlm_sample')
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        # if max_length is not None:
        #     warnings.warn(
        #         "`max_length` is deprecated in this function, use"
        #         " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
        #         UserWarning,
        #     )
        #     stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_cols = torch.ones((batch_size, self.num_heads), dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device) # [1, 2, ..., cur_len]

        generated_tokens_per_head = 0
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, insert_indices, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs, # input_ids, attention_mask, insert_indices, position_ids, cache_position, past_key_values, use_cache 
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            ) # batch size x tokens x vocab

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, insert_indices[1:], :] # batch_size x num_heads x vocab_size

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits) # logits processor does nothing []

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1) # batch_size x num_heads

            # finished columns should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                print('next_tokens', next_tokens.shape, 'unfinished_cols', unfinished_cols.shape)
                next_tokens = next_tokens * unfinished_cols + pad_token_id * (1 - unfinished_cols) # unfinished_cols==1 if col not previously had EOS

            # update generated ids, model inputs, and length for next step
            input_ids[:, insert_indices[1:]] = next_tokens
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None: # TODO per col
                unfinished_cols = next_tokens.ne(eos_token_id_tensor).long()

            # unfinished_cols = unfinished_cols & ~stopping_criteria(input_ids, scores) # stopping_criteria TODO col-based length check
            this_peer_finished = unfinished_cols.max() == 0
            
            generated_tokens_per_head += 1
            if generated_tokens_per_head == max_tokens_per_col: break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        if self.trace:
            print('in prepare inputs for generation with args', input_ids.shape, type(past_key_values), type(attention_mask), type(inputs_embeds), 
                type(cache_position), kwargs, sep='\n')
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache): # no because past_key_values is a tuple
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else: #here
                cache_length = past_length = past_key_values[0][0].shape[2] # number of toks in prev generation iter
                max_cache_length = None
            
            if self.trace:
                print('past_length', past_length, 'cache_length', cache_length, 'max_cache_length', max_cache_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                if self.trace:
                    print('input_ids case 1 before', input_ids.shape)
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                if self.trace:
                    print('input_ids case 1 after ', input_ids.shape)
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                if self.trace:
                    print('input_ids case 2 before', input_ids.shape)
                input_ids = input_ids[:, past_length:]
                if self.trace:
                    print('input_ids case 2 after ', input_ids.shape)
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if self.trace:
                print('position_ids', position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
            if self.trace:
                print('position_ids', position_ids)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None
            
        # insert_indices += torch.arange(0, insert_indices.shape[0]) # since ea col has one more token added to it

        model_inputs.update(
            {
                # "insert_indices": insert_indices,
                "position_ids": position_ids, # indices of input_ids to "look at"
                "cache_position": cache_position,
                "past_key_values": past_key_values, # from attention mechanism of headless LLamaModel
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask, #attention mask, cropped to only look at position_ids
            }
        )
        return model_inputs
    