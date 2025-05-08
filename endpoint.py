import modal
from dataclasses import asdict
from fastapi import Request

lora_volume = modal.Volume.from_name("vllm-loras", create_if_missing=True, environment_name="advay-dev")
models_volume = modal.Volume.from_name("models", create_if_missing=True, environment_name="advay-dev")

HF_CACHE_PATH = "/cache"
LORA_DIR = "/loras"
MODELS_DIR = "/models"
SCALEDOWN_WINDOW = 30
TIMEOUT = 60 * 60  # 1


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("uv")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_PATH, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"})  # faster model transfers
    .run_commands(
        "uv pip install --system vllm huggingface_hub[hf-xet] fastapi[standard]",
    )
    .entrypoint([])
)

app = modal.App("endpoint-test", image=image)

with image.imports():
    import logging

    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.serving_models import (
        BaseModelPath,
        OpenAIServingModels,
    )
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.protocol import CompletionRequest
    from vllm.lora.resolver import LoRARequest, LoRAResolver, LoRAResolverRegistry
    import torch
    from fastapi import Request


    class ConcreteLoRAModuleResolver(LoRAResolver):
        def __init__(self, base_model_id):
            self.base_model_id = base_model_id

        async def resolve_lora(self, base_model_name, lora_name) -> LoRARequest:
            lora_path = f"{LORA_DIR}/{lora_name}"
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_path=lora_path,
                lora_int_id=1,
            )
            return lora_request


@app.cls(volumes={
        MODELS_DIR: models_volume,
        LORA_DIR: lora_volume
    },
    image=image,
    gpu="H100!",
    min_containers=1,
    max_containers=1,
    cpu=4,
    memory=65536,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    region="us-chicago-1",
)
class Model:
    @modal.enter()
    async def enter(self):
        model = "Llama-3.1-8B"
        model_dir = f"{MODELS_DIR}/{model}"

        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model_dir,
                tokenizer_mode="auto",
                # enable_lora=True,
                max_loras=36,
                max_lora_rank=128,
                max_cpu_loras=500,
                tensor_parallel_size=torch.cuda.device_count(),
            )
        )

        model_config = await engine.get_model_config()
        base_model_path = [BaseModelPath(model, model_dir)]

        logging.info(f"model_config {model_config}")
        logging.info(f"base_model_path {base_model_path}")

        # lora_module_resolver = ConcreteLoRAModuleResolver(model)
        # LoRAResolverRegistry.register_resolver("lora-resolver", lora_module_resolver)
        openai_serving_models = OpenAIServingModels(
            engine,
            model_config,
            base_model_path,
            lora_modules=None,
            prompt_adapters=None,
        )
        self.openai_serving_completion = OpenAIServingCompletion(
            engine,
            model_config,
            openai_serving_models,
            request_logger=None,
        )

    @modal.fastapi_endpoint(method="POST")
    async def generate(self, request: Request):
        request_dict = await request.json()
        completion_request = CompletionRequest(**request_dict)
        response = await self.openai_serving_completion.create_completion(
            completion_request, request
        )

        return response