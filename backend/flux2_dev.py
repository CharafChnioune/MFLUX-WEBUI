"""Experimental FLUX.2-dev loader for MFLUX-WEBUI.

Upstream `mflux` (as of 0.16.3) ships Flux2 Klein (4B/9B) but not FLUX.2-dev.
The FLUX.2-dev checkpoint is also extremely large.

This module is intentionally self-contained and only used when the user
explicitly selects the `flux2-dev` alias.

Current approach:
- MLX transformer + VAE (reusing mflux Flux2 modules and weight mapping)
- Torch/Transformers Mistral3 text encoder for prompt embeddings

This is a pragmatic bridge until an all-MLX text encoder is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx import nn

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


# Keep in sync with diffusers
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/system_messages.py
SYSTEM_MESSAGE = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."

def _format_input(prompts: list[str], system_message: str = SYSTEM_MESSAGE):
    # Mirror diffusers formatting for the FLUX.2-dev text encoder.
    # Prompts are wrapped in a system+user conversation and tokenized via apply_chat_template.
    cleaned = [p.replace('[IMG]', '') for p in prompts]
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned
    ]


class Flux2DevWeightDefinition:
    """Only loads transformer + VAE from the diffusers-style FLUX.2-dev repo."""

    @staticmethod
    def get_transformer_mapping():
        # FLUX.2-dev ships extra guidance embedder weights (guidance distillation).
        # mflux's default Flux2 mapping targets the Klein models which have guidance_embeds
        # disabled, so we extend it here.
        mapping = list(Flux2WeightMapping.get_transformer_mapping())
        mapping.extend(
            [
                WeightTarget(
                    to_pattern="time_guidance_embed.guidance_linear_1.weight",
                    from_pattern=["time_guidance_embed.guidance_embedder.linear_1.weight"],
                ),
                WeightTarget(
                    to_pattern="time_guidance_embed.guidance_linear_2.weight",
                    from_pattern=["time_guidance_embed.guidance_embedder.linear_2.weight"],
                ),
            ]
        )
        return mapping

    @staticmethod
    def get_components():
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Flux2DevWeightDefinition.get_transformer_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers():
        return []


    @staticmethod
    def get_download_patterns():
        # Avoid the gigantic root-level merged `flux2-dev.safetensors` by only
        # requesting component subfolders.
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")


def _default_flux2_dev_model_config(model_name: str = "black-forest-labs/FLUX.2-dev") -> ModelConfig:
    # Based on `transformer/config.json` for black-forest-labs/FLUX.2-dev.
    return ModelConfig(
        priority=999,
        aliases=["flux2-dev"],
        model_name=model_name,
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 48,
            "num_attention_heads": 48,
            "attention_head_dim": 128,
            "joint_attention_dim": 15360,
            "guidance_embeds": True,
        },
        text_encoder_overrides={},
    )


class Flux2Dev(nn.Module):
    """txt2img for FLUX.2-dev (experimental).

    Warning: This checkpoint is huge. Initial loading requires downloading many
    gigabytes and can consume >100GB RAM unless you quantize.
    """

    vae: Flux2VAE
    transformer: Flux2Transformer

    def __init__(
        self,
        *,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        torch_device: str | None = None,
        torch_dtype: str | None = None,
        # Keep API-compatible kwargs (ignored for now)
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()

        self.prompt_cache = {}
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

        self.model_config = model_config or _default_flux2_dev_model_config()
        self._model_path = model_path
        self._torch_device = torch_device
        self._torch_dtype = torch_dtype

        path = model_path if model_path else self.model_config.model_name
        print(
            "[flux2-dev] Initializing experimental FLUX.2-dev loader. "
            "This model is extremely large and may take a long time to download/load."
        )
        weights = WeightLoader.load(weight_definition=Flux2DevWeightDefinition, model_path=path)

        self.vae = Flux2VAE()
        self.transformer = Flux2Transformer(**self.model_config.transformer_overrides)

        self.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Flux2DevWeightDefinition,
            models={
                "vae": self.vae,
                "transformer": self.transformer,
            },
        )

        self._torch_tokenizer = None
        self._torch_text_encoder = None

    def _ensure_torch_text_encoder(self):
        if self._torch_text_encoder is not None and self._torch_tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
        except Exception as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "FLUX.2-dev text encoder requires torch + transformers. "
                "Install them or use a Flux2 Klein model."
            ) from exc

        device = (self._torch_device or "cpu").strip()
        dtype_str = (self._torch_dtype or "bfloat16").strip().lower()
        if dtype_str in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16"}:
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_root = self.model_config.model_name
        if self._model_path:
            candidate = Path(self._model_path).expanduser()
            if candidate.exists() and candidate.is_dir() and (candidate / "tokenizer").exists() and (candidate / "text_encoder").exists():
                model_root = str(candidate)

        # Tokenizer lives in `tokenizer/` for FLUX.2-dev.
        self._torch_tokenizer = AutoTokenizer.from_pretrained(
            model_root,
            subfolder="tokenizer",
            use_fast=True,
        )
        if device == "mps" and dtype is torch.bfloat16:
            dtype = torch.float16

        self._torch_text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_root,
            subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if device != "cpu":
            self._torch_text_encoder.to(device)
        self._torch_text_encoder.eval()

    def _mistral3_out_layers(self) -> tuple[int, int, int]:
        """Pick 3 layer indices similar to Flux2 Klein's (9,18,27) on 36 layers.

        For 40 layers this becomes (10,20,30).
        """
        # Default from HF config: text_config.num_hidden_layers
        cfg = getattr(self._torch_text_encoder, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        num_layers = None
        if hasattr(text_cfg, "num_hidden_layers"):
            num_layers = int(text_cfg.num_hidden_layers)
        elif isinstance(text_cfg, dict) and "num_hidden_layers" in text_cfg:
            num_layers = int(text_cfg["num_hidden_layers"])
        elif hasattr(cfg, "num_hidden_layers"):
            num_layers = int(cfg.num_hidden_layers)
        if not num_layers:
            num_layers = 40
        return (num_layers // 4, num_layers // 2, (3 * num_layers) // 4)

    def _encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> tuple[mx.array, mx.array]:
        """Encode a prompt into FLUX.2 prompt embeddings.

        The FLUX.2-dev text encoder is chat-formatted (system+user messages) and expects
        tokenization via `apply_chat_template` (see diffusers' Flux2Pipeline).
        """
        self._ensure_torch_text_encoder()

        import torch

        tok = self._torch_tokenizer
        model = self._torch_text_encoder
        assert tok is not None
        assert model is not None

        # Mirror diffusers' `format_input` + `apply_chat_template` behavior.
        messages_batch = _format_input(prompts=[prompt], system_message=SYSTEM_MESSAGE)
        inputs = tok.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Mistral3 hidden_states are unavailable (output_hidden_states=True was ignored).")

        layer_ids = self._mistral3_out_layers()
        selected = [hidden_states[i] for i in layer_ids]
        stacked = torch.stack(selected, dim=1)  # (B, 3, S, H)
        b, n, s, h = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(b, s, n * h)

        # Convert to MLX.
        # NumPy does not support bfloat16; cast to float32 before converting.
        prompt_embeds = prompt_embeds.to(dtype=torch.float32)
        prompt_mx = mx.array(prompt_embeds.cpu().numpy()).astype(ModelConfig.precision)
        text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_mx)
        return prompt_mx, text_ids


    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, text_ids = self._encode_prompt(prompt)
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = self._encode_prompt(negative_prompt)
        return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 10,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 3.5,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
    ) -> GeneratedImage:
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )

        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=None,
            guidance=guidance,
        )

        latents, latent_ids, latent_height, latent_width = self._prepare_generation_latents(seed=seed, config=config)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        predict = self._predict(self.transformer)
        for t in config.time_steps:
            try:
                noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance=guidance,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _prepare_generation_latents(self, *, seed: int, config: Config) -> tuple[mx.array, mx.array, int, int]:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return Flux2LatentCreator.prepare_packed_latents(
                seed=seed,
                height=config.height,
                width=config.width,
                batch_size=1,
            )
        return self._prepare_img2img_latents(seed=seed, config=config)

    def _prepare_img2img_latents(self, *, seed: int, config: Config) -> tuple[mx.array, mx.array, int, int]:
        noise_latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )

        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        # Mirror Flux2 Klein edit helpers without importing them to keep this module small.
        if encoded.ndim == 3:
            encoded = mx.expand_dims(encoded, axis=0)
        if encoded.shape[-1] % 2 != 0 or encoded.shape[-2] % 2 != 0:
            encoded = encoded[:, :, : (encoded.shape[-2] // 2) * 2, : (encoded.shape[-1] // 2) * 2]
        encoded = self._match_latent_spatial_size(
            encoded=encoded,
            target_height=latent_height * 2,
            target_width=latent_width * 2,
        )

        encoded = Flux2LatentCreator.patchify_latents(encoded)
        # BN normalize (see Flux2 Klein edit): mean/var stats are part of VAE module.
        if hasattr(self.vae, "bn"):
            bn = getattr(self.vae, "bn")
            mean = getattr(bn, "running_mean", None)
            var = getattr(bn, "running_var", None)
            if mean is not None and var is not None:
                encoded = (encoded - mean.reshape(1, -1, 1, 1)) / mx.sqrt(var.reshape(1, -1, 1, 1) + 1e-5)

        clean_latents = Flux2LatentCreator.pack_latents(encoded)

        sigma = config.scheduler.sigmas[config.init_time_step]
        latents = LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=noise_latents, sigma=sigma)
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def _match_latent_spatial_size(*, encoded: mx.array, target_height: int, target_width: int) -> mx.array:
        _, _, height, width = encoded.shape
        if height != target_height:
            if height > target_height:
                offset = (height - target_height) // 2
                encoded = encoded[:, :, offset : offset + target_height, :]
            else:
                pad_total = target_height - height
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)))
        if width != target_width:
            if width > target_width:
                offset = (width - target_width) // 2
                encoded = encoded[:, :, :, offset : offset + target_width]
            else:
                pad_total = target_width - width
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)))
        return encoded

    @staticmethod
    def _predict(transformer):
        def predict(
            *,
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ):
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=guidance,
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=guidance,
                )
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        # Avoid mx.compile for FLUX.2-dev: the compiled graph can trigger Metal GPU timeouts on
        # large models. The uncompiled path is slower but more reliable.
        return predict
