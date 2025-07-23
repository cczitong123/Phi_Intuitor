# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

#新增
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model.path)#加入tokenizer

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

    
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", #     non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    #rollout 成功版，使用 combined_weights

    # @torch.no_grad()
    # def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    #     """
    #     多步 Phi-Decoding，batch 维度并行版
    #     """
    #     # —— 0. 原版 non_tensor_batch 处理 —— 
    #     non_tensor_batch = prompts.non_tensor_batch
    #     idx0 = prompts.batch["input_ids"]    # 用于生成 raw_prompt_ids 的基础
    #     bs0 = idx0.size(0)
    #     if "raw_prompt_ids" not in non_tensor_batch:
    #         non_tensor_batch["raw_prompt_ids"] = np.array(
    #             [_pre_process_inputs(self.pad_token_id, idx0[i]) for i in range(bs0)],
    #             dtype=object
    #         )
    #     if bs0 != len(non_tensor_batch["raw_prompt_ids"]):
    #         raise RuntimeError("vllm sharding manager is not work properly.")

    #     if "multi_modal_data" in non_tensor_batch:
    #         multi_modal_data = non_tensor_batch.pop("multi_modal_data")
    #         raw_prompt_ids   = non_tensor_batch.pop("raw_prompt_ids")
    #         vllm_inputs = [
    #             {"prompt_token_ids": r, "multi_modal_data": m}
    #             for r, m in zip(raw_prompt_ids, multi_modal_data)
    #         ]
    #     else:
    #         raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
    #         vllm_inputs    = [{"prompt_token_ids": r} for r in raw_prompt_ids]

    #     # —— 1. 超参数 —— 
    #     beam_size         = int(kwargs.get("step_beam_size",   self.config.step_beam_size))
    #     num_rollout       = int(kwargs.get("num_rollout",      self.config.num_rollout))
    #     num_foresight     = int(kwargs.get("num_foresight",    self.config.num_foresight))
    #     sigma_rate        = float(kwargs.get("sigma_rate",     self.config.sigma_rate))
    #     temperature       = float(kwargs.get("temperature",    self.config.temperature))
    #     step_response_len = int(kwargs.get("step_response_length", self.config.step_response_length))
    #     response_len      = int(kwargs.get("response_length",  self.config.response_length))

    #     #system_prompt = DEFAULT_SYSTEM_PROMPT

    #     # —— 2. 解码 raw prompts ——  
    #     idx         = prompts.batch["input_ids"]  # [bs, Lp]
    #     device      = idx.device
    #     bs          = idx.size(0)
    #     # 使用 raw_prompt_ids 解码文本
    #     raw_prompts    = self.tokenizer.batch_decode(idx, skip_special_tokens=True)

    #     # —— 3. SamplingParams for intermediate rollout ——  
    #     base_sp = SamplingParams(
    #         max_tokens=step_response_len,
    #         logprobs=1,
    #         temperature=temperature,
    #         n=num_rollout,
    #         stop=["\n", "<end_of_reasoning>"]
    #     )

    #     # —— 初始化 history/weight/log_probs ——  
    #     prev_steps      = [[""   for _ in range(beam_size)] for _ in range(bs)]
    #     prev_values     = [[0.0  for _ in range(beam_size)] for _ in range(bs)]
    #     weights_history = [[[]   for _ in range(beam_size)] for _ in range(bs)]
    #     rollout_log_probs = []  # <--- 初始化 rollout_log_probs

    #     # —— 4. 多步前瞻 (num_foresight) ——  
    #     for depth in range(num_foresight):
    #         all_inputs = []
    #         for b in range(bs):
    #             for k in range(beam_size):
    #                 prefix = (
    #                     f"User: {raw_prompts[b].strip()}\n"
    #                     f"Reasoning so far:\n{prev_steps[b][k]}"
    #                 )
    #                 all_inputs.append(prefix)
    #         prompt_ids = [
    #             self.inference_engine.llm_engine.tokenizer.encode(t, add_special_tokens=False)
    #             for t in all_inputs
    #         ]

    #         outs = self.inference_engine.generate(
    #             prompts=None,
    #             sampling_params=base_sp,
    #             prompt_token_ids=prompt_ids,
    #             use_tqdm=False
    #         )

    #         all_resp, all_lp = [], []
    #         for out in outs:
    #             for o in out.outputs:
    #                 txt = o.text.strip()
    #                 lp  = o.cumulative_logprob / (len(o.token_ids) + 1e-8)
    #                 all_resp.append(txt)
    #                 all_lp.append(lp)

    #         # 收集中间 log_probs
    #         rollout_log_probs.append(all_lp.copy())  # <--- 记录这一轮的 log_probs

    #         all_adv = []
    #         for b in range(bs):
    #             for k in range(beam_size):
    #                 base_idx = (b * beam_size + k) * num_rollout
    #                 prev_v   = prev_values[b][k]
    #                 for j in range(num_rollout):
    #                     all_adv.append(all_lp[base_idx + j] - prev_v)

    #         new_steps = [["" for _ in range(beam_size)] for _ in range(bs)]
    #         new_values = [[0.0 for _ in range(beam_size)] for _ in range(bs)]
    #         new_weights = [[[]  for _ in range(beam_size)] for _ in range(bs)]

    #         for b in range(bs):
    #             start = b * beam_size * num_rollout
    #             end   = start + beam_size * num_rollout
    #             resp_slice = all_resp[start:end]
    #             lp_slice   = all_lp[start:end]
    #             adv_slice  = np.array(all_adv[start:end])

    #             mu, sigma = np.mean(lp_slice), np.std(lp_slice)
    #             keep = [i for i,v in enumerate(lp_slice) if v > mu - sigma_rate * sigma]
    #             if len(keep) < beam_size:
    #                 wts = np.exp(adv_slice / temperature)
    #                 wts /= wts.sum()
    #                 extra = np.random.choice(
    #                     len(adv_slice),
    #                     beam_size - len(keep),
    #                     replace=False,
    #                     p=wts
    #                 ).tolist()
    #                 keep += extra
    #             keep.sort()

    #             adv_k = adv_slice[keep]
    #             combined_weights = softmax(adv_k / temperature)

    #             sel_idxs = np.random.choice(
    #                 len(keep), size=beam_size, replace=False, p=combined_weights
    #             ).tolist()
    #             picked = [keep[i] for i in sel_idxs]

    #             for k, sel in enumerate(picked):
    #                 origin_beam = sel // num_rollout
    #                 resp_text = resp_slice[sel]
    #                 raw_adv = float(adv_slice[sel])
    #                 token_ids = self.inference_engine.llm_engine.tokenizer.encode(resp_text, add_special_tokens=False)
    #                 rep_adv = [raw_adv] * len(token_ids)

    #                 new_weights[b][k] = weights_history[b][origin_beam] + rep_adv
    #                 new_steps[b][k]   = prev_steps[b][origin_beam] + resp_text + "\n"
    #                 new_values[b][k]  = lp_slice[sel]

    #         prev_steps, prev_values, weights_history = new_steps, new_values, new_weights
    #     #print(f"prev_steps: {prev_steps}")###check prev_steps

    #     # —— 6. 最后一轮并行生成最终答案 ——(argmax)
    #     final_prompts = []
    #     history_list  = []
    #     final_rewards = []
    #     final_probs   = []
    #     for b in range(bs):
    #         # # 选择最大的 prev_value 对应的 beam
    #         # best_k = int(np.argmax(prev_values[b]))
    #         # —— 基于 prev_values 重新计算增益权重（adv）并做 softmax 采样 —— 
    #         vals = np.array(prev_values[b])                     # shape: [beam_size]
    #         adv  = vals - vals.mean()                           # centered advantage
    #         beam_p = np.exp(adv / temperature)
    #         beam_p /= beam_p.sum()                                # 归一化概率
    #         best_k = int(np.random.choice(len(beam_p), p=beam_p))  # 按概率采样一个 beam
    #         p = float(beam_p[best_k])   # 把选中那条 beam 的概率取出来
    #         #新adv####
    #         adv_all = vals - vals.mean()####
    #         raw_adv_final = float(adv_all[best_k])####

    #         history = prev_steps[b][best_k]
    #         base_ws   = weights_history[b][best_k]
    #         txt = (
    #             #f"{system_prompt}\n\n"
    #             f"User: {raw_prompts[b].strip()}\n"
    #             f"Reasoning so far:\n{prev_steps[b][best_k]}\n"
    #             #f"Final step: Continue the reasoning above—include **all** intermediate steps—and write out the full reasoning process, "
    #             #f"Final step: ending with the answer in the format \\boxed{{…}} and finish the reasoning with <end_of_reasoning>."
    #         )
    #         #print(f"raw_prompts[{b}]:{raw_prompts[b]}")###check raw_prompts
    #         #print(f"prev_steps[{b}]:{prev_steps[b][best_k]}")###check prev_steps
    #         #print(f"final_prompt[{b}]:{txt}")###check final_prompt
    #         ids = self.tokenizer.encode(txt, add_special_tokens=False)
    #         # 重复 n 次，保证 final_prompts 和 history_list 都是 bs * n 长度
    #         #for _ in range(self.config.n):
    #         final_prompts.append(ids)
    #         history_list.append(history)
    #         final_rewards.append(list(base_ws))
    #             #final_probs.append(p)  # 保存选中 beam 的概率
    #         final_probs.append(raw_adv_final)####adv版

    #     final_sp = SamplingParams(
    #         max_tokens = response_len,
    #         logprobs   = 1,
    #         temperature= temperature,
    #         n          = 1,
    #         stop       = ["<end_of_reasoning>"]
    #     )
    #     final_outs = self.inference_engine.generate(
    #         prompts=None,
    #         sampling_params=final_sp,
    #         prompt_token_ids=final_prompts,
    #         use_tqdm=False
    #     )

        
    #     full_texts = []
    #     final_rewards_padded = []
    #     for i, (history, out) in enumerate(zip(history_list, final_outs)):
    #         gen_text = out.outputs[0].text.strip()
    #         full_texts.append(history + gen_text)
    #         token_ids = self.tokenizer.encode(gen_text, add_special_tokens=False)
    #         #final_rewards_padded.append(final_rewards[i] + [1.0] * len(token_ids))
    #         prob = final_probs[i]
    #         final_rewards_padded.append(final_rewards[i] + [prob] * len(token_ids))

    #     # encode & pad responses
    #     full_resp_ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in full_texts]
    #     #full_resp_ids = [ids[:response_len] for ids in full_resp_ids]#截断到 response_len
    #     resp_padded = pad_2d_list_to_length(
    #         full_resp_ids,
    #         self.tokenizer.pad_token_id,
    #         max_length=response_len
    #     ).to(device)

    #     # # —— 8. 构建 prm_reward 张量 ——
    #     # 使得 prm_reward 的每行长度与 resp_padded 的响应长度一致 (response_len)
    #     pr_tensors = []
    #     for r in final_rewards_padded:
    #         # 截断或补齐到 response_len
    #         if len(r) >= response_len:
    #             row = r[:response_len]
    #         else:
    #             row = r + [0.0] * (response_len - len(r))
    #         pr_tensors.append(row)
    #     prm_reward = torch.tensor(pr_tensors, device=device)
    #     ####neu reward 这样计算导致reward过小
    #     # # 按公式算权重：w_i = exp(-r_i/T) / sum_j exp(-r_j/T)
    #     # r = prm_reward
    #     # T = temperature 
    #     # exp_neg = torch.exp(-r / T)           # [Bn, L]
    #     # den = exp_neg.sum(dim=1, keepdim=True)  # [Bn, 1]
    #     # w = exp_neg / den                       # [Bn, L]

    #     # # 4) 最终 r*_i = w_i * r_i
    #     # r_star = w * r                          # [Bn, L]

    #     # # 5) 用 r_star 作为 prm_reward
    #     # prm_reward = r_star

    #     # —— 9. repeat 原 prompt tensors & 拼 batch —— repeat 原 prompt tensors & 拼 batch ——
    #     Bn = resp_padded.size(0)
    #     repeat = Bn // bs
    #     idx    = prompts.batch["input_ids"]#.repeat_interleave(repeat, dim=0)
    #     mask   = prompts.batch["attention_mask"]#.repeat_interleave(repeat, dim=0)
    #     pos    = prompts.batch["position_ids"]#.repeat_interleave(repeat, dim=0)
    #     seq    = torch.cat([idx, resp_padded], dim=1)
    #     delta  = torch.arange(1, response_len+1, device=device).unsqueeze(0).expand(Bn, -1)
    #     last_p = pos[:, -1:].expand(-1, response_len)
    #     pos    = torch.cat([pos, last_p+delta], dim=1)
    #     attn   = get_response_mask(resp_padded, self.tokenizer.eos_token_id, mask.dtype)
    #     mask   = torch.cat([mask, attn], dim=1)

    #     batch = TensorDict({
    #         "prompts":        idx,
    #         "responses":      resp_padded,
    #         "input_ids":      seq,
    #         "attention_mask": mask,
    #         "position_ids":   pos,
    #         #"prm_reward":     prm_reward,
    #     }, batch_size=Bn)
    #     print(f"[DEBUG] batch['prompts'].shape: {batch['prompts'].shape}")####check prompts shape
    #     print(f"[DEBUG] batch['responses'].shape: {batch['responses'].shape}")####check responses shape
    #     print(f"[DEBUG] batch['input_ids'].shape: {batch['input_ids'].shape}")####check input_ids shape
    #     print(f"[DEBUG] batch['attention_mask'].shape: {batch['attention_mask'].shape}")####check attention_mask shape
    #     print(f"[DEBUG] batch['position_ids'].shape: {batch['position_ids'].shape}")####check position_ids shape
    #     print(f"[DEBUG] prm_reward.shape: {prm_reward.shape}")

    #     # print(f"[DEBUG] resp_padded.shape: {resp_padded.shape}")####check resp_padded shape
    #     # print(f"[DEBUG] prm_reward.shape: {prm_reward.shape}")####check prm_reward shape
    #     # print(f"[DEBUG] prm_reward[0]: {prm_reward[0]}")####check prm_reward[0]
    #     # with open("prm_reward3_0.txt", "w", encoding="utf-8") as f:
    #     #     f.write(str(prm_reward[0].tolist()))
    #     # # 将第一个样本的 response 文本保存到文件
    #     # first_resp_ids = resp_padded[0].tolist()
    #     # #print(f"[DEBUG] first_resp_ids: {first_resp_ids}")####check first_resp_ids
    #     # first_resp_text = self.tokenizer.decode(first_resp_ids, skip_special_tokens=True)
    #     # print(f"[DEBUG] first_resp_text: {first_resp_text}")####check first_resp_text
    #     # with open("first_response3_0.txt", "w", encoding="utf-8") as f:
    #     #     f.write(first_resp_text)
    #     # # 可选：打印文件路径以确认
    #     # print("[DEBUG] Saved prm_reward to prm_reward_0.txt and first response to first_response_0.txt")
    #     #return DataProto(batch=batch)
        
    #     rollout_log_probs = pad_2d_list_to_length(
    #                 rollout_log_probs, -1, max_length=self.config.response_length
    #             ).to(idx.device)
    #     rollout_log_probs = rollout_log_probs.to(torch.float32)

    #     print(f"[DEBUG] rollout_log_probs.shape: {rollout_log_probs.shape}")####check rollout_log_probs shape
        
    #     # 展开 rollout_log_probs 至新的 batch_size
    #         #expanded_rlp = rollout_log_probs.repeat_interleave(repeat, dim=0) 
    #     #batch["rollout_log_probs"] = rollout_log_probs
    #     #print(f"[DEBUG] batch keys: {list(batch.keys())}")####check batch keys
        

    #     # 原来的 non_tensor_batch
    #     old_ntb = non_tensor_batch
    #     new_ntb = {}

    #     # 计算重复次数
    #     repeat = 1 #Bn // bs  # 1024 // 256 = 4

    #     for k, v in old_ntb.items():
    #         if isinstance(v, list):
    #             # 普通 Python 列表，直接把整个列表 repeat 次 then flatten
    #             new_ntb[k] = list(itertools.chain.from_iterable([v] * repeat))
    #         elif isinstance(v, np.ndarray):
    #             # numpy 数组，先把第一维 expand，再 reshape
    #             # 若 v.shape=(bs, ...)，np.repeat 会把第 0 维 repeat 倍
    #             new_ntb[k] = np.repeat(v, repeat, axis=0)
    #         elif isinstance(v, torch.Tensor):
    #             # 如果有 Tensor，也可以用 repeat_interleave
    #             new_ntb[k] = v.repeat_interleave(repeat, dim=0)
    #         else:
    #             # 标量或其他不可重复类型，就简单复制同一个值 repeat 次
    #             new_ntb[k] = [v] * Bn

    #     # 最终调用 DataProto 时，传入展开后的 non_tensor_batch：
    #     return DataProto(
    #         batch=batch,
    #         non_tensor_batch=new_ntb  
    #     )


    #     #return DataProto(batch=batch,, non_tensor_batch=non_tensor_batch) #


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
