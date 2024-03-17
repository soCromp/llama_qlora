from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM, get_peft_config, LoraModel
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers.generation.utils import GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import List, Optional, Tuple, Union
import torch
import json
import os

# structure: Peft model (Qlora) > Lora model (base_model)  > Normal model > Headless version of normal model
# ohmodel: PeftModelForCausalLM             > LoraModel > LlamaForCausalLM          > LlamaModel
# mhmodel: MultiHeadPeftModelForCausalLM    > LoraModel > MultiheadLlamaForCausalLM > LlamaModel

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]


class MultiheadLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, model, heads, config):
        super().__init__(config['llamamodel_config'])
        self.model = model # LlamaModel
        self.heads = heads
        self.lm_head = heads[0]
        self.post_init()
        
    
    def from_other(othermodel, orighead, masks, vocabs, config):
        assert len(masks) == len(vocabs)
        return MultiheadLlamaForCausalLM(othermodel, [orighead], config)
    
    
    def _add_head(self, orighead):
        """in features and out features the same, but a mask will be applied at prediction.
        Basically just a method to deep-copy the original head"""
        in_features, out_features, bias = orighead.in_features, orighead.out_features, orighead.bias
        newhead = type(orighead)(in_features, out_features, bias)
        newhead.load_state_dict(orighead.state_dict())
        self.heads.append(newhead)
        
        
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Copied from LlamaForCausalLM forward()
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

        hidden_states = outputs[0] #.to(input_ids.device)
        print(hidden_states.shape)
        # self.lm_head = self.lm_head.to(hidden_states.device)
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
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
        
        
    def can_generate(self): return True
    
    
    def generate(self, *args, **kwargs):
        """Generates output tokens. args empty; kwargs includes input_ids and attention_mask"""
        # self.lm_head = self.heads[0].cuda(0)
        return super().generate(*args, **kwargs)
    
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
            Copied from GenerationMixin
            Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
            used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
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
                [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
                `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.
            ```"""
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
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
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
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens.to(unfinished_sequences.device) * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


class MultiHeadPeftModelForCausalLM(PeftModelForCausalLM):
    def __init__(self, config, base_model):
        super().__init__(base_model, config) #sets self.base_model to base_model
        self.config = config
    
    
    def from_one_head(peft_model, head_masks, head_max_tokens, *args, **kwargs):
        """peft_model: A PeftModelForCausalLM,
            head_masks: a list of masks for head output vocabulary,
            head_max_tokens: a list of ints, where the nth element is the 
                        max number of new tokens to be generated by the nth head"""
        headless_model = peft_model.base_model.model.model #LlamaModel
        orighead = peft_model.base_model.model.lm_head
        normalmodel_config = {} 
        normalmodel_config['llamamodel_config'] = peft_model.base_model.model.config
        
        normalmodel = MultiheadLlamaForCausalLM.from_other(headless_model, orighead, [], [], normalmodel_config) # replaces LlamaForCausalLM        
        mhmodel = MultiHeadPeftModelForCausalLM(peft_model.peft_config['default'], normalmodel)
        return mhmodel
        
        
    def generate(self, *args, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
            
        outputs = self.base_model.generate(*args, **kwargs)
        return outputs
        
        