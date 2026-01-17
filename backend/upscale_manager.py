import os
import gc
import time
import random
import traceback
import mlx.core as mx
from PIL import Image
import numpy as np
from pathlib import Path
import json
import tempfile
from backend.mflux_compat import Config, ModelConfig
from backend.mlx_utils import force_mlx_cleanup, print_memory_usage
from backend.flux_manager import parse_scale_factor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
_UPSCALE_MODEL = None
UPSCALE_STEPS = int(os.getenv("MFLUX_UPSCALE_STEPS", "12"))
UPSCALE_STRENGTH = float(os.getenv("MFLUX_UPSCALE_STRENGTH", "0.75"))
UPSCALE_QUANTIZE = os.getenv("MFLUX_UPSCALE_QUANTIZE")


def _get_quantize_value():
    if UPSCALE_QUANTIZE is None or UPSCALE_QUANTIZE == "":
        return None
    try:
        return int(UPSCALE_QUANTIZE)
    except ValueError:
        return None


def _get_upscale_model():
    """
    Lazily load the Flux ControlNet upscaler. Fails quietly and allows PIL fallback.
    """
    global _UPSCALE_MODEL
    if _UPSCALE_MODEL is not None:
        return _UPSCALE_MODEL
    try:
        # Import lazily so environments without controlnet support still run with PIL fallback.
        try:
            from mflux.controlnet.flux_controlnet import Flux1Controlnet
        except ModuleNotFoundError:
            from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet  # type: ignore

        _UPSCALE_MODEL = Flux1Controlnet(
            model_config=ModelConfig.dev_controlnet_upscaler(),
            quantize=_get_quantize_value(),
            local_path=None,
            lora_paths=None,
            lora_scales=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Upscaler unavailable, falling back to PIL resize: {exc}")
        _UPSCALE_MODEL = None
    return _UPSCALE_MODEL

def upscale_image(image_path, upscale_factor=2):
    """
    Upscale an image using a local resampling upscaler (LANCZOS).
    """
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    new_w = int(w * upscale_factor)
    new_h = int(h * upscale_factor)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def upscale_image_gradio(
    input_image, upscale_factor, output_format, metadata
):
    """
    Upscale an image using the Flux ControlNet upscaler when available,
    falling back to high-quality PIL resize when not.
    """
    try:
        # Parse inputs
        if not input_image:
            return None, "Error: Input image is required"
        
        # Get original image dimensions
        original_image = Image.open(input_image)
        original_width, original_height = original_image.width, original_image.height
        
        # Parse upscale factor (can be scale factor like "2x" or absolute value)
        if isinstance(upscale_factor, str) and upscale_factor.strip():
            # Try to parse as scale factor
            if upscale_factor.strip().lower().endswith('x'):
                try:
                    scale = float(upscale_factor.strip().lower().replace('x', ''))
                    upscale_factor_int = int(scale)
                except ValueError:
                    upscale_factor_int = 2
            else:
                try:
                    upscale_factor_int = int(float(upscale_factor))
                except ValueError:
                    upscale_factor_int = 2
        else:
            upscale_factor_int = int(upscale_factor) if upscale_factor else 2
        
        # Validate upscale factor
        if upscale_factor_int not in [2, 3, 4]:
            return None, "Error: Upscale factor must be 2, 3, or 4"

        target_w = int(original_width * upscale_factor_int)
        target_h = int(original_height * upscale_factor_int)

        # Try high-quality Flux upscaler first
        upscaled_image = None
        status_detail = None
        flux_model = _get_upscale_model()

        if flux_model:
            try:
                generated = flux_model.generate_image(
                    seed=int(time.time()),
                    prompt="High quality detailed image",
                    controlnet_image_path=input_image,
                    config=Config(
                        num_inference_steps=UPSCALE_STEPS,
                        height=target_h,
                        width=target_w,
                        controlnet_strength=UPSCALE_STRENGTH,
                    ),
                )
                upscaled_image = generated.image
                status_detail = "Flux ControlNet upscaler"
            except Exception as exc:  # noqa: BLE001
                status_detail = f"Upscaler unavailable, falling back to PIL resize: {exc}"
                print(status_detail)
                upscaled_image = None

        if upscaled_image is None:
            print(f"Upscaling image with factor {upscale_factor_int}x using PIL")
            upscaled_image = upscale_image(input_image, upscale_factor_int)
            if upscaled_image is None:
                return None, "Error: Failed to upscale image"
            status_detail = status_detail or "PIL LANCZOS resize"
            
        # Save the upscaled image
        timestamp = int(time.time())
        
        # Determine file extension
        if output_format == "PNG":
            ext = "png"
            save_kwargs = {"format": "PNG"}
        elif output_format == "JPEG":
            ext = "jpg"
            save_kwargs = {"format": "JPEG", "quality": 95}
        else:  # WebP
            ext = "webp"
            save_kwargs = {"format": "WebP", "quality": 95}
            
        filename = f"upscaled_{upscale_factor_int}x_{timestamp}.{ext}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        upscaled_image.save(output_path, **save_kwargs)
        
        # Save metadata if requested
        if metadata:
            metadata_path = os.path.join(OUTPUT_DIR, f"upscaled_{upscale_factor_int}x_{timestamp}.json")
            
            # Get original image info
            original_image = Image.open(input_image)
            
            metadata_dict = {
                "original_width": original_image.width,
                "original_height": original_image.height,
                "upscaled_width": upscaled_image.width,
                "upscaled_height": upscaled_image.height,
                "upscale_factor": upscale_factor_int,
                "output_format": output_format,
                "generation_time": str(time.ctime()),
                "original_file": os.path.basename(input_image)
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=2)
                
        print(f"Upscaled image saved to {output_path}")
        
        # Return both the image and a success message
        info_message = f"Successfully upscaled image {upscale_factor_int}x to {upscaled_image.width}x{upscaled_image.height}"
        if status_detail:
            info_message = f"{info_message} ({status_detail})"
        return upscaled_image, info_message
        
    except Exception as e:
        print(f"Error in upscaling: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"
        
    finally:
        # Cleanup
        gc.collect()
        force_mlx_cleanup()

# Expose this module as `upscale_manager`
import sys as _sys
upscale_manager = _sys.modules[__name__]

def upscale_with_custom_dimensions_gradio(
    input_image, target_width, target_height, output_format, metadata
):
    """
    Upscale an image to custom dimensions using scale factors or absolute values.
    """
    try:
        if not input_image:
            return None, "Error: Input image is required"
        
        # Get original image dimensions
        original_image = Image.open(input_image)
        original_width, original_height = original_image.width, original_image.height
        
        # Parse target dimensions
        final_width = parse_scale_factor(target_width, original_width)
        final_height = parse_scale_factor(target_height, original_height)
        
        # Calculate effective scale factor
        scale_x = final_width / original_width
        scale_y = final_height / original_height
        
        # Use the larger scale factor to maintain aspect ratio
        effective_scale = max(scale_x, scale_y)
        
        # Round to nearest supported scale factor
        if effective_scale <= 2.5:
            upscale_factor_int = 2
        elif effective_scale <= 3.5:
            upscale_factor_int = 3
        else:
            upscale_factor_int = 4
        
        # Upscale the image
        print(f"Upscaling image with factor {upscale_factor_int}x to target {final_width}x{final_height}")
        upscaled_image = upscale_image(input_image, upscale_factor_int)
        
        if upscaled_image is None:
            return None, "Error: Failed to upscale image"
        
        # Resize to exact target dimensions if needed
        if upscaled_image.width != final_width or upscaled_image.height != final_height:
            upscaled_image = upscaled_image.resize((final_width, final_height), Image.Resampling.LANCZOS)
        
        # Save the upscaled image
        timestamp = int(time.time())
        
        # Determine file extension
        if output_format == "PNG":
            ext = "png"
            save_kwargs = {"format": "PNG"}
        elif output_format == "JPEG":
            ext = "jpg"
            save_kwargs = {"format": "JPEG", "quality": 95}
        else:  # WebP
            ext = "webp"
            save_kwargs = {"format": "WebP", "quality": 95}
            
        filename = f"upscaled_custom_{final_width}x{final_height}_{timestamp}.{ext}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        upscaled_image.save(output_path, **save_kwargs)
        
        # Save metadata if requested
        if metadata:
            metadata_path = os.path.join(OUTPUT_DIR, f"upscaled_custom_{final_width}x{final_height}_{timestamp}.json")
            
            metadata_dict = {
                "original_width": original_width,
                "original_height": original_height,
                "upscaled_width": upscaled_image.width,
                "upscaled_height": upscaled_image.height,
                "target_width": target_width,
                "target_height": target_height,
                "effective_scale": effective_scale,
                "upscale_factor_used": upscale_factor_int,
                "output_format": output_format,
                "generation_time": str(time.ctime()),
                "original_file": os.path.basename(input_image)
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=2)
                
        print(f"Upscaled image saved to {output_path}")
        
        # Return both the image and a success message
        info_message = f"Successfully upscaled image to {upscaled_image.width}x{upscaled_image.height}"
        return upscaled_image, info_message
        
    except Exception as e:
        print(f"Error in custom upscaling: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"
        
    finally:
        # Cleanup
        gc.collect()
        force_mlx_cleanup()

def batch_upscale_gradio(
    input_images, upscale_factor, output_format, metadata
):
    """
    Batch upscale multiple images.
    """
    try:
        if not input_images:
            return [], "Error: No images provided"
            
        upscaled_images = []
        errors = []
        
        for idx, image_path in enumerate(input_images):
            try:
                print(f"Processing image {idx+1}/{len(input_images)}")
                
                # Upscale individual image
                upscaled, message = upscale_image_gradio(
                    image_path, upscale_factor, output_format, metadata
                )
                
                if upscaled:
                    upscaled_images.append(upscaled)
                else:
                    errors.append(f"Image {idx+1}: {message}")
                    
            except Exception as e:
                errors.append(f"Image {idx+1}: {str(e)}")
                
        if upscaled_images:
            success_msg = f"Successfully upscaled {len(upscaled_images)} image(s)"
            if errors:
                success_msg += f"\n\nErrors:\n" + "\n".join(errors)
            return upscaled_images, success_msg
        else:
            return [], "Failed to upscale any images\n\n" + "\n".join(errors)
            
    except Exception as e:
        print(f"Error in batch upscaling: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}"

def _resolve_image_path(input_image):
    """
    Accepts a PIL image, file-like object, or path-like input and returns a tuple of
    (path_to_image, temp_file_path). If a temp file is created, caller should clean it up.
    """
    temp_path = None

    # Gradio may pass PIL.Image or a dict with a "name" field; handle both.
    if isinstance(input_image, Image.Image):
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as f:
            input_image.save(f, format="PNG")
        image_path = temp_path
    elif isinstance(input_image, dict) and input_image.get("name"):
        image_path = input_image["name"]
    elif hasattr(input_image, "name"):
        image_path = input_image.name
    else:
        image_path = input_image

    return image_path, temp_path


def upscale_single_gradio(
    input_image,
    prompt=None,
    model_name="dev",
    target_width=None,
    target_height=None,
    seed=None,
    guidance=None,
    steps=None,
    controlnet_strength=None,
    vae_tiling=False,
    vae_tiling_split="horizontal",
    save_metadata=True,
    low_ram_mode=False,
    quantize=None,
):
    """
    Compatibility wrapper used by the Gradio Upscale tab.
    Accepts the wider parameter set but delegates to the lightweight upscaler.
    """
    temp_path = None
    try:
        image_path, temp_path = _resolve_image_path(input_image)

        # Ensure we have sensible target dimensions; fall back to the source image size.
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
        target_width = target_width or orig_w
        target_height = target_height or orig_h

        result_image, info = upscale_with_custom_dimensions_gradio(
            input_image=image_path,
            target_width=target_width,
            target_height=target_height,
            output_format="PNG",
            metadata=save_metadata,
        )
        return result_image, info
    except Exception as e:
        print(f"Error in upscale_single_gradio: {str(e)}")
        traceback.print_exc()
        return None, f"Error: {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def upscale_batch_gradio(
    image_files,
    prompt=None,
    model_name="dev",
    scale_factor="2x",
    seed=None,
    guidance=None,
    steps=None,
    controlnet_strength=None,
    vae_tiling=False,
    vae_tiling_split="horizontal",
    save_metadata=True,
    low_ram_mode=False,
    quantize=None,
    progress_callback=None,
):
    """
    Batch upscale wrapper to match the Gradio expectations.
    Uses a single scale factor for all images and returns a list of upscaled PIL images.
    """
    if not image_files:
        return []

    results = []
    for idx, img_input in enumerate(image_files):
        if progress_callback:
            try:
                progress_callback(idx)
            except Exception:
                pass

        img_path, tmp_path = _resolve_image_path(img_input)
        try:
            with Image.open(img_path) as img:
                target_w = parse_scale_factor(scale_factor, img.width)
                target_h = parse_scale_factor(scale_factor, img.height)

            upscaled, _ = upscale_with_custom_dimensions_gradio(
                input_image=img_path,
                target_width=target_w,
                target_height=target_h,
                output_format="PNG",
                metadata=save_metadata,
            )
            if upscaled:
                results.append(upscaled)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return results
