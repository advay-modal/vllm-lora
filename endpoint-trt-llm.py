import asyncio
import hashlib
import json
from pathlib import Path
import os
import time
from uuid import uuid4

from fastapi.responses import JSONResponse
from fastapi import Request

import modal

modal_start_time = time.monotonic()  # on remote, time that code started running Modal

here = Path(__file__).parent
deployment_id = uuid4()

MINUTES = 60  # seconds

CLOUD = "oci"

app_name = "openpipe-trtllm-latency-test"
app = modal.App(app_name)

volume = modal.Volume.from_name(f"{app_name}-volume", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
REMOTE_DEFAULT_CONFIG_PATH = "/default.json"
REMOTE_BS4_CONFIG_PATH = "/bs4.json"


def get_system_prompt(model_id):
    if "qwen" in model_id.lower():
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    elif "llama" in model_id.lower():
        return "You are a helpful, harmless, and honest AI assistant created by Meta."
    elif "writer" in model_id.lower() and "med" in model_id.lower():
        return "You are a highly knowledgeable and experienced expert in the healthcare and biomedical field, possessing extensive medical knowledge and practical expertise."
    else:
        return False

tensorrt_image = (
    modal.Image.from_registry(
        # "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.10
    )
    .entrypoint(
        []  # remove verbose logging by base image on entry
    )
    .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
    .pip_install(
        "tensorrt-llm==0.19.0",
        "pynvml", 
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install(
        "flashinfer-python",
        extra_index_url="https://flashinfer.ai/whl/cu126/torch2.6/",
    )
    .pip_install(
        "hf-transfer==0.1.9",
        "huggingface_hub==0.31.3",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(MODELS_PATH),
        }
    )
    .add_local_file(
        here / "configs" / "openpipe" / "default.json",
        REMOTE_DEFAULT_CONFIG_PATH
    )
    .add_local_file(
        here / "configs" / "openpipe" / "bs4.json",
        REMOTE_BS4_CONFIG_PATH
    )
)

with tensorrt_image.imports():
    from dataclasses import asdict
    from huggingface_hub import snapshot_download
    import numpy as np
    import random
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams, BuildConfig
    from tensorrt_llm.executor import LoRARequest
    from tensorrt_llm.llmapi import (
        QuantConfig,
        KvCacheConfig,
        CalibConfig,
        LookaheadDecodingConfig,
    )
    from tensorrt_llm.lora_manager import LoraConfig
    from tensorrt_llm.plugin.plugin import PluginConfig
    import torch

loras_volume = modal.Volume.from_name("openpipe-loras", create_if_missing=True, environment_name="luis-dev")
LORAS_PATH = Path("/loras")

@app.cls(
    image=tensorrt_image,
    scaledown_window=20 * MINUTES,
    min_containers=1,
    max_containers=1,
    gpu="H100",
    timeout=60 * MINUTES,
    cpu=4,
    memory=4 * 1024,
    cloud=CLOUD,
    volumes={
        VOLUME_PATH: volume,
        LORAS_PATH: loras_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class AsyncTRTLLMEngineService:
    def construct_engine_kwargs(self) -> dict:
        """Convert engine kwargs string into config objects"""

        print("creating trtllm config objects")
        config_name_to_constructor = {
            "quant_config": lambda x: QuantConfig(**x),
            "kv_cache_config": lambda x: KvCacheConfig(**x),
            "calib_config": lambda x: CalibConfig(**x),
            "build_config": lambda x: BuildConfig(**x),
            "plugin_config": PluginConfig.from_dict,
            "lora_config": lambda x: LoraConfig(**x),
            # TODO: Add support for other decoding configs
            "speculative_config": lambda x: LookaheadDecodingConfig(**x),
        }

        engine_kwargs = json.loads(self.engine_kwargs_string)
        config_hash = hash_config_string(json.dumps(engine_kwargs, sort_keys=True))

        def construct_configs(x):
            for key, value in x.items():
                if type(value) is dict:
                    x[key] = construct_configs(value)
                if key in config_name_to_constructor:
                    x[key] = config_name_to_constructor[key](value)
            return x

        engine_kwargs = construct_configs(engine_kwargs)

        self.lookahead_config = engine_kwargs.get("speculative_config")
        self.lora_request = None
        if engine_kwargs.get("lora_config"):
            self.lora_request = [
                    LoRARequest(f"lora-{i}", i, a_lora_dir)
                    for i, a_lora_dir in enumerate(engine_kwargs["lora_config"].lora_dir)
            ]

        engine_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        print("number of GPUs:", engine_kwargs["tensor_parallel_size"])

        return config_hash, engine_kwargs

    def build_engine(self, engine_path, engine_kwargs) -> None:
        """Build a new TRT-LLM engine and save it to a volume"""

        print(f"building a new trtllm engine at {engine_path}")
        llm = LLM(model=self.model_path, **engine_kwargs)
        llm.save(engine_path)
        llm.shutdown()
        del llm

    @modal.enter()
    def enter(self):
        """Load the TRT-LLM engine onto the GPU and prepare for inference"""
        from transformers import AutoTokenizer

        # self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_id = "NousResearch/Meta-Llama-3.1-8B-Instruct"
        # self.engine_kwargs_string = Path(REMOTE_DEFAULT_CONFIG_PATH).read_text()
        self.engine_kwargs_string = Path(REMOTE_BS4_CONFIG_PATH).read_text()
        self.model_path = MODELS_PATH / self.model_id

        seed_everything()

        print("downloading base model if necessary")
        snapshot_download(self.model_id, local_dir=self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        config_hash, engine_kwargs = self.construct_engine_kwargs()

        engine_path = (
            self.model_path / f"{tensorrt_llm.__version__}_engine" / config_hash
        )
        if not os.path.exists(engine_path):
            self.build_engine(engine_path, engine_kwargs)

        print(f"loading engine from {engine_path}")
        self.llm = LLM(model=engine_path, **engine_kwargs)

        self.cold_boot_s = time.monotonic() - modal_start_time

        self.generate_impl("foo", {"temperature": 0.8, "top_p": 0.95})

    async def generate_impl(self, prompt: str, sampling_params_kwargs: dict) -> dict:
        assert "lookahead_config" not in sampling_params_kwargs
        sampling_params = SamplingParams(
            **sampling_params_kwargs,
            lookahead_config=self.lookahead_config,
        )

        if get_system_prompt(self.model_id) and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": get_system_prompt(self.model_id)},
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        start_time = time.monotonic()
        output = await self.llm.generate_async(
            text, sampling_params, lora_request=self.lora_request 
        )
        llm_latency_ms = int(1000 * (time.monotonic() - start_time))

        outputs = [asdict(output.outputs[0])]  # generation samples
        results = {
            "stats": {
                "llm_latency_ms": llm_latency_ms,
                "cold_boot_s": self.cold_boot_s,
            },
            "outputs": outputs,
        }

        return results

    @modal.method()
    async def xgenerate(self, prompt: str, sampling_params_kwargs: dict) -> dict:
        return await self.generate_impl(prompt, sampling_params_kwargs)

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request) -> str:
        sampling_params_kwargs = {
            "temperature":0.8,
            "top_p":0.95,
            "max_tokens":96,  # max generated tokens
        }

        request_dict = await request.json()
        results = await self.generate_impl(
            request_dict['prompt'], sampling_params_kwargs
        )

        output = results["outputs"][0]
        return JSONResponse(content=output)


    @modal.exit()
    def exit(self):
        print("container exiting, shutting down trtllm engine")
        self.llm.shutdown()
        del self.llm


@app.local_entrypoint()
async def main(
    prompt_key: str = "prompt",
    output_key: str = "output",
    max_prompts: int = None,
    save_results_path: str = here / "outputs" / "results.jsonl",
):
    config_path: str = here / "configs" / "openpipe" / REMOTE_BS4_CONFIG_PATH[1:]
    config = json.loads(Path(config_path).read_text())
    model_id = config["model_id"]
    dataset_path: str = config["dataset_path"]
    engine_type = config["engine_type"]
    engine_kwargs = config["engine_kwargs"]
    sampling_kwargs = config["sampling_kwargs"]
    infrastructure_kwargs = config["infrastructure_kwargs"]
    concurrency_kwargs = config["concurrency_kwargs"]

    experiment_id = generate_experiment_id()
    print(f"running experiment {experiment_id} with {model_id} and hyperparameters:")
    pretty_print_config(
        engine_type,
        json.dumps(engine_kwargs, indent=4),
        json.dumps(sampling_kwargs, indent=4),
        json.dumps(infrastructure_kwargs | concurrency_kwargs, indent=4),
    )

    print("building modal llm service class")
    if engine_type == "trtllm":
        engine_kwargs["tensor_parallel_size"] = get_gpu_count(
            infrastructure_kwargs["gpu"]
        )
        model_class = (AsyncTRTLLMEngineService
            .with_options(**infrastructure_kwargs)
            .with_concurrency(**concurrency_kwargs)
        )
    else:
        assert 0

    engine_kwargs_string = json.dumps(
        engine_kwargs
    )  # *** AFTER tensor_parallel_size ***
    model_service = model_class()

    print("loading dataset... ", end="")
    prompts, oracle_outputs = load_dataset(
        Path(dataset_path),
        prompt_key,
        output_key,
        max_prompts,
    )
    print(f"with {len(prompts)} prompts")

    num_cold_prompts = get_num_cold_prompts(
        infrastructure_kwargs, concurrency_kwargs
    )
    print(f"warming up containers with {num_cold_prompts} cold prompts")
    function_calls = [
        model_service.xgenerate.spawn("warm", sampling_kwargs)
        for _ in range(num_cold_prompts)
    ]
    modal.FunctionCall.gather(*function_calls)

    start_time = time.monotonic()
    print(f"spawning {len(prompts)} function calls")
    tasks = [
        fetch(prompt, sampling_kwargs, oracle_output, model_service.xgenerate)
        for prompt, oracle_output in zip(prompts, oracle_outputs)
    ]

    print("collecting results")
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        result["engine_type"] = engine_type
        result["engine_kwargs"] = engine_kwargs
        result["sampling_kwargs"] = sampling_kwargs
        result["infrastructure_config"] = infrastructure_kwargs
        result["concurrency_kwargs"] = concurrency_kwargs
        result["experiment_id"] = experiment_id
        results.append(result)
    total_latency_s = time.monotonic() - start_time
    print(f"Total latency for all requests: {total_latency_s:.2f}s")

    save_results(save_results_path, results)

    print(f"finished experiment {experiment_id} with {model_id} and hyperparameters:")
    pretty_print_config(
        engine_type,
        json.dumps(engine_kwargs, indent=4),
        json.dumps(sampling_kwargs, indent=4),
        json.dumps(infrastructure_kwargs | concurrency_kwargs, indent=4),
    )
    latencies = [r["stats"]["llm_latency_ms"] for r in results]
    cold_boots_s = list(set([round(r["stats"]["cold_boot_s"], 2) for r in results]))
    print(f"\t cold boots: {cold_boots_s}")

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)

    print(f"\t inference latency (p50, p90): ({p50:.2f}ms, {p90:.2f}ms)")
    print(f"\t inference latencies = {latencies}")
    print("")
    print(
        f"\t output lengths={[len(r['outputs'][0]['token_ids']) for r in results[:]]}"
    )
    breakpoint()


def get_gpu_count(gpu_string):
    if ":" not in gpu_string:
        return 1
    return int(gpu_string[gpu_string.index(":") + 1 :])


async def fetch(prompt, sampling_kwargs, oracle_output, target):
    start_time = time.monotonic()
    out_handle = target.spawn(prompt, sampling_kwargs)
    result = {
        "prompt_text_length": len(prompt),
        "prompt": prompt,
        "oracle_output": oracle_output,
        "out_handle": out_handle,
        "start_time": start_time,
    }
    response = await result["out_handle"].get.aio()
    end_time = time.monotonic()
    result |= response

    result["client_latency_ms"] = int(1000 * (end_time - start_time))

    result["response_token_count"] = sum(
        len(output["token_ids"]) for output in response["outputs"]
    )
    result["out_handle"] = result["out_handle"].object_id
    result["deployment_id"] = str(deployment_id)

    return result


def load_dataset(path, prompt_key, output_key, max_prompts):
    prompts = []
    outputs = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj[prompt_key] if prompt_key else obj)
            outputs.append(obj.get(output_key))
            if max_prompts and len(prompts) >= max_prompts:
                break

    return prompts, outputs


def get_num_cold_prompts(infrastructure_kwargs, concurrency_kwargs):
    if "max_containers" not in infrastructure_kwargs:
        return 1

    ci = concurrency_kwargs["max_inputs"] or 1
    return (ci * (infrastructure_kwargs["max_containers"] - 1)) + 1


def save_results(path, results):
    with open(path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_experiment_id():
    from wonderwords import RandomWord

    r_gen = RandomWord()
    return "-".join(
        [
            r_gen.word(
                include_parts_of_speech=[x], word_min_length=4, word_max_length=7
            )
            for x in ["verb", "adjective", "noun"]
        ]
    )


def hash_config_string(config_string):
    return hashlib.md5(config_string.encode()).hexdigest()


def pretty_print_config(
    engine_type,
    engine_kwargs_json_string,
    sampling_kwargs_json_string,
    infra_concurrency_kwargs_json_string,
):
    engine_lines = engine_kwargs_json_string.splitlines()
    sampling_lines = sampling_kwargs_json_string.splitlines()
    infra_concurrency_lines = infra_concurrency_kwargs_json_string.splitlines()

    print(f"{(engine_type + ' ENGINE CONFIG').center(55)} | ", end="")
    print(f"{'SAMPLING CONFIG'.center(30)} | ", end="")
    print(f"{'INFRA + CONCURRENCY CONFIG'.center(30)}")
    for ll in range(
        max(len(engine_lines), len(sampling_lines), len(infra_concurrency_lines))
    ):
        engine_line = engine_lines[ll] if ll < len(engine_lines) else ""
        sampling_line = sampling_lines[ll] if ll < len(sampling_lines) else ""
        infra_concurrency_line = (
            infra_concurrency_lines[ll] if ll < len(infra_concurrency_lines) else ""
        )

        print(
            f"{engine_line[4:]:<55} | {sampling_line[4:]:<30} | {infra_concurrency_line[4:]:<30}"
        )
