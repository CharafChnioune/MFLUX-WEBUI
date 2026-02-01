"""
LoRA key converter for FLUX.2 Klein models.

Converts LoRA files from diffusion_model.* format (ComfyUI/SimpleTuner)
to base_model.model.* format (mflux compatible).
"""

import re
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


def convert_lora_keys(input_path: str, output_path: str | None = None) -> str:
    """
    Convert LoRA keys from diffusion_model.* to mflux-compatible format.

    Handles:
    - diffusion_model.double_blocks.X.img_attn.* -> base_model.model.double_blocks.X.img_attn.*
    - diffusion_model.double_blocks.X.txt_attn.* -> base_model.model.double_blocks.X.txt_attn.*
    - diffusion_model.double_blocks.X.img_mlp.* -> lora_unet_double_blocks_X_img_mlp_*
    - diffusion_model.double_blocks.X.txt_mlp.* -> lora_unet_double_blocks_X_txt_mlp_*
    - diffusion_model.single_blocks.X.* -> base_model.model.single_blocks.X.*

    Args:
        input_path: Path to input LoRA file
        output_path: Path for output file (default: input_path with _mflux suffix)

    Returns:
        Path to converted file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_mflux{input_path.suffix}"
    else:
        output_path = Path(output_path)

    # Load weights
    tensors = {}
    with safe_open(str(input_path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Convert keys
    converted = {}
    for key, tensor in tensors.items():
        new_key = key

        # Handle img_mlp and txt_mlp patterns - convert to lora_unet_ format
        # diffusion_model.double_blocks.0.img_mlp.0.lora_A.weight -> lora_unet_double_blocks_0_img_mlp_0.lora_down.weight
        mlp_match = re.match(
            r"diffusion_model\.double_blocks\.(\d+)\.(img_mlp|txt_mlp)\.(\d+)\.lora_([AB])\.weight",
            key
        )
        if mlp_match:
            block_num = mlp_match.group(1)
            mlp_type = mlp_match.group(2)
            mlp_idx = mlp_match.group(3)
            lora_type = "down" if mlp_match.group(4) == "A" else "up"
            new_key = f"lora_unet_double_blocks_{block_num}_{mlp_type}_{mlp_idx}.lora_{lora_type}.weight"
        # Handle attention patterns - convert to base_model.model format
        elif key.startswith("diffusion_model."):
            new_key = key.replace("diffusion_model.", "base_model.model.", 1)

        converted[new_key] = tensor

    # Save converted weights
    save_file(converted, str(output_path))

    print(f"Converted {len(converted)} keys")
    print(f"Saved to: {output_path}")

    return str(output_path)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lora_converter.py <input_lora.safetensors> [output_lora.safetensors]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_lora_keys(input_path, output_path)


if __name__ == "__main__":
    main()
