from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

import modal

models_volume = modal.Volume.from_name("models", create_if_missing=True)
loras_volume = modal.Volume.from_name("openpipe-loras", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# need to set "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY" annd "AWS_REGION" in the secret
s3_access_credentials = modal.Secret.from_name("cloud-bucket-mount-secret", environment_name="main")
s3_bucket_mount = modal.CloudBucketMount("modal-s3mount-test-bucket", secret = s3_access_credentials)


MODELS_DIR = "/models/hub"
LORAS_DIR = "/loras"
S3_DIR = "/loras_s3"

image = (
    modal.Image.debian_slim(python_version="3.12")
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
        "uv pip install --system huggingface_hub[hf_transfer]==0.30.0",
        "uv pip install --system vllm==0.8.5.post1",
        "uv pip install --system flashinfer-python==0.2.2.post1 -i https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .env({"VLLM_CONFIGURE_LOGGING": "0", "TORCH_HOME": "/compile_cache"})
)

app = modal.App("openpipe-lora-v1", image=image)

with image.imports():
    import logging
    import shutil
    import os
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
            s3_path = (Path(S3_DIR) / lora_name).as_posix()
            if not os.path.exists(lora_path):
                if not os.path.exists(s3_path):
                    raise ValueError(f"LoRA {lora_name} not found in {LORAS_DIR} or {S3_DIR}")
                # Copy from S3 to local volume if not present
                shutil.copytree(s3_path, lora_path)
            lora_request = LoRARequest(lora_name=lora_name, lora_path=lora_path, lora_int_id=abs(hash(lora_name)))
            return lora_request


@app.cls(
    volumes={MODELS_DIR: models_volume, LORAS_DIR: loras_volume, "/compile_cache": vllm_cache, S3_DIR: s3_bucket_mount},
    image=image,
    gpu="H100!",
    min_containers=1,
    buffer_containers=1, # keeps one container on standby in case load spikes
    scaledown_window=5 * 60,
    timeout=5 * 65,
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
                enable_lora=True,
                max_loras=4,
                max_lora_rank=8,
            )
        )

        model_config = await engine.get_model_config()
        base_model_path = [BaseModelPath(model, MODELS_DIR)]

        logging.info(f"model_config {model_config}")
        logging.info(f"base_model_path {base_model_path}")

        # Load LoRAs from a Modal Volume.
        lora_module_resolver = LoRaResolverFromVolume(model)
        LoRAResolverRegistry.register_resolver("lora-resolver", lora_module_resolver)

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