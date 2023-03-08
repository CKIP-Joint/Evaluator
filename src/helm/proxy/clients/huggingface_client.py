import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List
import os
import threading

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence, wrap_batched_request_time
from .huggingface_tokenizer import HuggingFaceTokenizers

os.environ["TOKENIZERS_PARALLELISM"] = "0"
TOKENIZER = {}

def get_tokenizer(model_name):
    _id = threading.get_ident()
    tokenizer = TOKENIZER.get(id, None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        TOKENIZER[_id] = tokenizer
    return tokenizer

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class HuggingFaceServer:
    def __init__(self, model_name: str):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"

        with htrack_block("Loading model"):
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        with htrack_block("Loading tokenizer"):
            #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = get_tokenizer(model_name)

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt").to(self.device)

        raw_request["do_sample"] = False
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        raw_request["output_hidden_states"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(raw_request["stop_sequences"])
            # Total number of stop words should be 1.
            assert len(stop_sequence_ids.input_ids) == 1
            # Total number of tokens in each stop word should be 1.
            assert len(stop_sequence_ids.input_ids[0]) == 1
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # Use HuggingFace's `generate` method.
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        # TODO: Get rid of the extra tokenization?
        all_tokens = [self.tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        completions = []
        for (decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts) in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}
    
    def serve_batched_request(self, batched_raw_request: Dict[str, Any]):
        # Note: batched_raw_request is similar to raw_request. The only difference is `prompt` is a list of prompt str
        ''' santiy check
        result = []
        for raw_request_prompt in batched_raw_request['prompt']:
            raw_request = {**batched_raw_request, 'prompt': raw_request_prompt}
            result.append(self.serve_request(raw_request))

        return result
        '''
        batched_encoded_input = self.tokenizer(batched_raw_request["prompt"], return_tensors="pt", padding=True).to(self.device)

        raw_request = {**batched_raw_request, 'prompt': ""}
        raw_request["do_sample"] = False
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        raw_request["output_hidden_states"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(raw_request["stop_sequences"])
            # Total number of stop words should be 1.
            assert len(stop_sequence_ids.input_ids) == 1
            # Total number of tokens in each stop word should be 1.
            assert len(stop_sequence_ids.input_ids[0]) == 1
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # Use HuggingFace's `generate` method.
        # The output shape:
        #   1. batch size = 1, greedy search, num_return_sequences=1: output.sequences=(1, seq_len)
        #   2. batch size = N, greedy search, num_return_sequences=1: output.sequences=(N, seq_len)
        #   3. batch size = N, greedy search, num_return_sequences=M: output.sequences=(N*M, seq_len) [seq1_1, seq1_2, ..., seq1_M, seq2_1, ...]
        batched_output = self.model.generate(**batched_encoded_input, **relevant_raw_request)
        batched_sequences = batched_output.sequences
        batched_scores = batched_output.scores

        results = []
        # TODO please make this more efficient. Changle just use a for loop 
        # to process output like the batchsize=1 case
        for seq_id in range(0, batched_sequences.shape[0], raw_request["num_return_sequences"]):
            seq_ids = list(range(seq_id, seq_id+raw_request["num_return_sequences"]))
            sequences = batched_sequences[seq_ids, :]
            scores = [score[seq_ids,:] for score in batched_scores]
            encoded_input = {
                "input_ids":batched_encoded_input.input_ids[seq_ids, :],
                "attention_mask": batched_encoded_input.attention_mask[seq_ids, :],
            }
            encoded_input = dotdict(encoded_input)

            # Compute logprobs for each completed sequence.
            all_logprobs_of_chosen_tokens = []
            all_top_logprobs_dicts = []
            for completion_id in range(raw_request["num_return_sequences"]):
                logprobs_of_chosen_tokens = []
                top_logprobs_dicts = []
                for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                    logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                    # Get top tokens in terms of log probability.
                    topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                    top_logprobs_dicts.append(
                        {
                            self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                            for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                        }
                    )

                    # Get log probability of chosen token.
                    j = i + len(encoded_input.input_ids[0])
                    logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
                all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
                all_top_logprobs_dicts.append(top_logprobs_dicts)

            # Remove prompt from the start of each sequence if echo_prompt is False.
            if not raw_request["echo_prompt"]:
                sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

            # TODO: Get rid of the extra tokenization?
            all_tokens = [self.tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
            all_decoded_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

            completions = []
            for (decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts) in zip(
                all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
            ):
                completions.append(
                    {
                        "text": decoded_text,
                        "tokens": tokens,
                        "logprobs": logprobs_of_chosen_tokens,
                        "top_logprobs_dicts": top_logprobs_dicts,
                    }
                )

            results.append({"completions": completions, "input_length": len(encoded_input.input_ids[0])})

        return results

class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}
        self.batchable = True

    def get_model_server_instance(self, model_engine) -> HuggingFaceServer:
        if model_engine not in self.model_server_instances:
            if model_engine == "gpt-j-6b":
                self.model_server_instances[model_engine] = HuggingFaceServer("EleutherAI/gpt-j-6B")
            elif model_engine == "gpt2":
                self.model_server_instances[model_engine] = HuggingFaceServer("gpt2")
            elif model_engine == "bloom":
                self.model_server_instances[model_engine] = HuggingFaceServer("bloom")
            
            # HF BLOOM 
            elif model_engine == "bloom_1b1":
                self.model_server_instances[model_engine] = HuggingFaceServer("/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloom-1B1/snapshots/1e718be072aa40714c9dc34e35ea6b64979a65ad")
            elif model_engine == "bloom_3b":
                self.model_server_instances[model_engine] = HuggingFaceServer("/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloom-3B/snapshots/515ae965cc83b9ebbf0054de106c434bd4ec35dc")
            elif model_engine == "bloom_7b1":
                self.model_server_instances[model_engine] = HuggingFaceServer("/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloom-7B1/snapshots/850ba1758a7744fedae78caadc152625133b1677")
            
            # HF BLOOMZ
            elif model_engine == "bloomz_1b1":
                self.model_server_instances[model_engine] = HuggingFaceServer("/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloomz-1B1")
            elif model_engine == "bloomz_3b":
                self.model_server_instances[model_engine] = HuggingFaceServer("/proj/gpu_d_98001/huggingface/hub/models--bigscience--bloomz-3B")
            # HF Codegen
            elif model_engine == "codegen_2b_nl":
                self.model_server_instances[model_engine] = HuggingFaceServer( "/proj/gpu_d_98001/huggingface/hub/models--Salesforce--codegen-2B-nl/snapshots/5e06b1a66d42091757983a5b549ff100743f3904")
            
            # TO-BE-EVALUATED
            elif model_engine == "to_be_evaluated":
                import argparse
                parser = argparse.ArgumentParser()
                parser.add_argument("--model_id", default=None)
                parser.add_argument("--load_path", default=None)
                args, unknown = parser.parse_known_args()
                if args.load_path:
                    self.model_server_instances[model_engine] = HuggingFaceServer(args.load_path)
                elif os.environ.get('EVALUATED_MODEL_PATH'):
                    self.model_server_instances[model_engine] = HuggingFaceServer(os.environ.get('EVALUATED_MODEL_PATH'))
                else:
                    raise Exception(f"--load_path is not specified!\nCannot find model in EVALUATED_MODEL_PATH: {os.environ.get('EVALUATED_MODEL_PATH')}!")
            else:
                raise Exception("Unknown model!")
        return self.model_server_instances[model_engine]

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model_engine)

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            #"stop_sequences": request.stop_sequences, # Changle's comment: I don't know why stop sequence is specified by scenario instead of model
            "stop_sequences": [model_server_instance.tokenizer.eos_token],
        }

        try:

            def do_it():
                return model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )


    def make_batched_request(self, batched_request: List[Request]):
        if len(batched_request)==1:
            return [self.make_request(batched_request[0])]
        
        # checking 
        for request in batched_request:
            if request.embedding:
                return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

            # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
            if len(request.stop_sequences) > 1:
                raise ValueError("More than one stop sequence is not supported.")

        # Get cached model server instance if possible (to save on model and tokenizer loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(batched_request[0].model_engine)

        # get batchced prompt
        batched_prompt = [request.prompt for request in batched_request]
        batched_raw_request = {
            "engine": batched_request[0].model_engine,
            "prompt": batched_prompt,
            "temperature": 1e-7 if batched_request[0].temperature == 0 else batched_request[0].temperature,
            "num_return_sequences": batched_request[0].num_completions,
            "max_new_tokens": batched_request[0].max_tokens,
            "top_p": batched_request[0].top_p,
            "echo_prompt": batched_request[0].echo_prompt,
            "top_k_per_token": batched_request[0].top_k_per_token,
            #"stop_sequences": request.stop_sequences, # Changle's comment: I don't know why stop sequence is specified by scenario instead of model
            "stop_sequences": [model_server_instance.tokenizer.eos_token],
        }

        try:
            # changle: No need to use cache mechanism here
            def do_it():
                return model_server_instance.serve_batched_request(batched_raw_request)
            batched_response = wrap_batched_request_time(do_it)()

        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        batched_request_result = []
        for response in batched_response:
            completions = []
            for raw_completion in response["completions"]:
                sequence_logprob: float = 0
                tokens: List[Token] = []

                if request.echo_prompt:
                    # Add prompt to list of generated tokens.
                    generated_tokens = raw_completion["tokens"][response["input_length"] :]
                    for token_text in raw_completion["tokens"][: response["input_length"]]:
                        tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
                else:
                    generated_tokens = raw_completion["tokens"]

                # Compute logprob for the entire sequence.
                for token_text, logprob, top_logprobs_dict in zip(
                    generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
                ):
                    tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                    sequence_logprob += logprob

                completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
                completion = truncate_sequence(completion, request)
                completions.append(completion)

            request_result= RequestResult(
                                            success=True,
                                            cached=False,
                                            request_time=response["request_time"],
                                            request_datetime=response.get("request_datetime"),
                                            completions=completions,
                                            embedding=[],
                                        )

            batched_request_result.append(request_result)
        
        return batched_request_result


    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    tokens = tokenizer.tokenize(request.text)
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
