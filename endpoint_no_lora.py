from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

import modal

lora_volume = modal.Volume.from_name("vllm-loras", create_if_missing=True, environment_name="advay-dev")
models_volume = modal.Volume.from_name("models", create_if_missing=True, environment_name="advay-dev")
loras_volume = modal.Volume.from_name("openpipe-loras", create_if_missing=True, environment_name="luis-dev")
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MODELS_DIR = "/models/hub"
LORAS_DIR = "/loras"


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODELS_DIR,
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "VLLM_INSTALL_PUNICA_KERNELS": "1",
        }
    )
    .run_commands(
        "uv pip install --system huggingface_hub[hf-xet] fastapi[standard]",
    )
    .run_commands("uv pip install --system torch==2.6.0")
    .run_commands("uv pip install --system flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/")
    .run_commands("uv pip install --system flash-attn --no-build-isolation")
    .run_commands("uv pip install --system hf_transfer")
    .apt_install("build-essential", "cmake", "clang")
    .run_commands("uv pip install --system sentencepiece --no-build-isolation")
    .apt_install("rustc", "cargo")
    .apt_install("libomp-dev")
    .run_commands("uv pip install --system xformers==0.0.29.post2 --no-build-isolation")
    .run_commands(
        "apt remove rustc cargo -y",
        "apt install curl -y",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .apt_install("python3-dev", "build-essential", "pkg-config", "libssl-dev")
    .run_commands('export PATH="$HOME/.cargo/bin:$PATH" && uv pip install --system vllm==v0.8.5.post1')
    .env({"VLLM_CONFIGURE_LOGGING": "0", "TORCH_HOME": "/compile_cache"})
    .entrypoint([])
)

app = modal.App("openpipe-no-lora", image=image)

with image.imports():
    import logging

    import torch
    from fastapi import Request
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_models import (
        BaseModelPath,
        OpenAIServingModels,
    )
    from vllm.lora.resolver import LoRARequest, LoRAResolver, LoRAResolverRegistry

    class LoRaResolverFromVolume(LoRAResolver):
        """Resolve LoRAs from a Modal Volume. Modal Volumes are optimized for caching,
        improving performance when compared to other remote file locations like S3."""

        def __init__(self, base_model_id):
            self.base_model_id = base_model_id

        async def resolve_lora(self, _base_model_name, lora_name) -> LoRARequest:
            lora_path = (Path(LORAS_DIR) / lora_name).as_posix()
            lora_request = LoRARequest(lora_name=lora_name, lora_path=lora_path, lora_int_id=abs(hash(lora_name)))
            return lora_request


@app.cls(
    volumes={MODELS_DIR: models_volume, LORAS_DIR: loras_volume, "/compile_cache": vllm_cache},
    image=image,
    gpu="H100!",
    min_containers=1,
    max_containers=1,
    cpu=4,
    memory=65_536,
    scaledown_window=5 * 60,
    timeout=5 * 65,
    region="us-chicago-1",
    secrets=[modal.Secret.from_name("huggingface-secret", environment_name="main")],
)
@modal.concurrent(max_inputs=1000)
class vLLMLora:
    @modal.enter()
    async def enter(self):
        model = "meta-llama/Llama-3.1-8B-Instruct"

        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                tokenizer_mode="auto",
                dtype="float16",
                enable_chunked_prefill=True,
                max_num_partial_prefills=16,  # fewer concurrent prefills → lower P99
                # long_prefill_token_threshold=1024,
                # max_long_partial_prefills=2,
                max_num_batched_tokens=2048,
                max_num_seqs=32,
                enforce_eager=False,
                max_model_len=1024,
                gpu_memory_utilization=0.80,
                enable_prefix_caching=True,
                tensor_parallel_size=torch.cuda.device_count(),
            )
        )

        model_config = await engine.get_model_config()
        base_model_path = [BaseModelPath(model, MODELS_DIR)]

        logging.info(f"model_config {model_config}")
        logging.info(f"base_model_path {base_model_path}")

        # Load LoRAs from a Modal Volume.

        openai_serving_models = OpenAIServingModels(
            engine,
            model_config,
            base_model_path,
            prompt_adapters=None,
        )
        self.openai_serving_completion = OpenAIServingCompletion(
            engine,
            model_config,
            openai_serving_models,
            request_logger=None,
        )

    @modal.fastapi_endpoint(method="POST", custom_domains=["op-llama3-lora.modal-endpoints.com"])
    async def generate(self, request: Request):
        request_dict = await request.json()
        completion_request = CompletionRequest(**request_dict)
        generator = await self.openai_serving_completion.create_completion(completion_request, request)

        if isinstance(generator, ErrorResponse):
            logging.info(f"generator {generator}")
            return JSONResponse(status_code=500, content=generator.model_dump())
        elif completion_request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return generator.model_dump()