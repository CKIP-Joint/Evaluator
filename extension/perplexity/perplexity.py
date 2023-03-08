# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perplexity Metric."""

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_bloom_inference.inference_server.model_handler import ModelDeployment
from transformers_bloom_inference.inference_server.utils import get_argument_parser, parse_args, print_rank_n
import argparse
import math
from tqdm import tqdm

def get_tbi_model_and_tokenizer(model_name):
    args = {
        "deployment_framework": "hf_accelerate_exposed",
        "model_name": model_name,
        "model_class": "AutoModelForCausalLM",
        "dtype": 'auto',
        "generate_kwargs": {
            "min_length": 2, 
            "max_new_tokens": 100, 
            "do_sample": False,
        },
        "max_input_length": None,
        "max_batch_size": None,
    }
    model = ModelDeployment(argparse.Namespace(**args), True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

## This is hf's perplexity method
class Perplexity():

    def compute(
        self, 
        list_of_text_str, 
        model_name, 
        batch_size: int = 16, 
        add_start_token: bool = True, 
        device=None, 
        max_length=None
    ):


        model, tokenizer = get_tbi_model_and_tokenizer(model_name)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            list_of_text_str,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss = []
        token_num = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model.model.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            
            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()
        
        #return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
            
            loss_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            token_num_batch = shift_attention_mask_batch.sum(1)

            loss += loss_batch.tolist()
            token_num += token_num_batch.tolist()

        return {"perplexity": np.exp(np.sum(loss)/np.sum(token_num)),"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


# This is Megatron's perplexity method
class Perplexity_Megatron():
    seq_len = 2048
    hop = 32
    overalapping_eval = 32
    tokenizer = None
    model = None
    tokens = None
    
    def get_many_seq(self):
        seqs = []
        loss_masks = []
        atten_masks = []
        total_targets = len(self.tokens) - 1
        targets = max(total_targets - self.overalapping_eval, 0)
        self.total_sequences = max(
            math.ceil(targets / self.overalapping_eval) + 1, 1)
        for idx in range(self.total_sequences):
            item = self.__getitem__(idx)
            seqs.append(item['text'])
            loss_masks.append(item['pad_mask'])
            atten_masks.append(item['atten_mask'])

        seqs = torch.stack([torch.from_numpy(seq) for seq in seqs])
        loss_masks = torch.stack([torch.from_numpy(loss_mask) for loss_mask in loss_masks])
        atten_masks = torch.stack([torch.from_numpy(atten_mask) for atten_mask in atten_masks])
        return seqs.to('cuda'), loss_masks.to('cuda'), atten_masks.to('cuda')

    # copied from megatron's lm
    def __getitem__(self, idx):
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx + 1]
        atten_mask = self.attn_masks[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        pad_mask = [1] * num_tokens
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.tokenizer.pad_token_id] * num_pad
            atten_mask += [1] * num_pad
        pad_mask = np.array(pad_mask[1:])
        if self.overalapping_eval != self.seq_len and idx != 0:
            pad_mask[:-self.overalapping_eval] *= 0

        return {'text': np.array(tokens), 'pad_mask': pad_mask, 'atten_mask': np.array(atten_mask)}

    def compute(
        self, 
        all_content_in_one_str, 
        model_name, 
        batch_size: int = 16, 
        add_start_token: bool = True, 
        device=None, 
        max_length=None
    ):
        self.model, self.tokenizer = get_tbi_model_and_tokenizer(model_name)
        encodings = self.tokenizer(
            all_content_in_one_str,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        self.tokens = encodings["input_ids"]
        self.attn_masks = encodings["attention_mask"]
        encoded_texts, loss_masks, attn_masks = self.get_many_seq()        

        ppls = []
        loss = []
        token_num = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]
            loss_mask = loss_masks[start_index:end_index]

            labels = encoded_batch
            with torch.no_grad():
                out_logits = self.model.model.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
            shift_loss_mask_batch = loss_mask[...].contiguous()

            loss_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch * shift_loss_mask_batch).sum(1)
            perplexity_batch = torch.exp(
                loss_batch
                / shift_attention_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()
        
        #return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
            
            loss += loss_batch.tolist()

        return {"perplexity": np.exp(np.sum(loss)/len(self.tokens)),"perplexities": ppls, "mean_perplexity": np.mean(ppls)}




## This is DS's perplexity method
class DS_Perplexity():
    seq_len = 2048

    # TODO
    # You needs to mask the padded part of the last sequence during loss calcualtion
    def get_many_seq(self, seq, attention_mask=False):
        seqs = []
        for idx in range(0, seq.shape[1], self.seq_len):
            seg = seq[0, idx: min(idx+self.seq_len, seq.shape[1])]
            if seg.shape[0] < self.seq_len:
                num_pad = self.seq_len - seg.shape[0]
                if attention_mask:
                    seg = torch.cat([seg, torch.tensor([0] * num_pad).type(seg.dtype).to(seg.device)])
                else:    
                    seg = torch.cat([seg, torch.tensor([self.tokenizer.pad_token_id] * num_pad).type(seg.dtype).to(seg.device)])
            seqs.append(seg)
        seqs = torch.stack(seqs, dim=0)
        return seqs

    def compute(
        self, 
        all_content_in_one_str, 
        model_name, 
        batch_size: int = 16, 
        add_start_token: bool = True, 
        device=None, 
        max_length=None
    ):


        self.model, self.tokenizer = get_tbi_model_and_tokenizer(model_name)

        encodings = self.tokenizer(
            all_content_in_one_str,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = self.get_many_seq(encodings["input_ids"])
        attn_masks = self.get_many_seq(encodings["attention_mask"], attention_mask=True)
        

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss = []
        token_num = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model.model.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            
            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()
        
        #return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
            
            loss_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            token_num_batch = shift_attention_mask_batch.sum(1)

            loss += loss_batch.tolist()
            token_num += token_num_batch.tolist()

        return {"perplexity": np.exp(np.sum(loss)/np.sum(token_num)),"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
