import os
import json
import subprocess
import tempfile
import traceback
import threading
import queue
from PIL import Image
from pathlib import Path

def prepare_training_config(
    base_model,
    trigger_word,
    epochs,
    batch_size,
    lora_rank,
    output_dir,
    transformer_blocks_enabled,
    transformer_start,
    transformer_end,
    single_blocks_enabled,
    single_start,
    single_end,
    image_size,
    tmpdir
):
    """
    Prepare the training configuration.
    """
    target_size = int(image_size.split('x')[0])
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "model": "dev" if "dev" in base_model else "schnell",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4 if base_model.endswith("-4-bit") else (8 if base_model.endswith("-8-bit") else None),
        "width": target_size,
        "height": target_size,
        "training_loop": {
            "num_epochs": int(epochs),
            "batch_size": int(batch_size)
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-4
        },
        "save": {
            "output_path": os.path.abspath(output_dir),
            "checkpoint_frequency": 10
        },
        "instrumentation": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
            "validation_prompt": f"{trigger_word}"
        },
        "lora_layers": {},
        "examples": {
            "path": tmpdir,
            "images": []
        }
    }
    
    if transformer_blocks_enabled:
        config["lora_layers"]["transformer_blocks"] = {
            "block_range": {
                "start": int(transformer_start),
                "end": int(transformer_end)
            },
            "layer_types": [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out"
            ],
            "lora_rank": int(lora_rank)
        }

    if single_blocks_enabled:
        config["lora_layers"]["single_transformer_blocks"] = {
            "block_range": {
                "start": int(single_start),
                "end": min(int(single_end), 20)
            },
            "layer_types": [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v"
            ],
            "lora_rank": int(lora_rank)
        }

    return config

def resize_image(input_path, output_path, target_size):
    """
    Resize an image while maintaining aspect ratio.
    """
    with Image.open(input_path) as img:
        width, height = img.size
        ratio = min(target_size / width, target_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        resized.save(output_path)

def run_training(
    base_model,
    uploaded_files,
    trigger_word,
    epochs,
    batch_size,
    lora_rank,
    output_dir,
    resume_checkpoint,
    mlx_vlm_model,
    transformer_blocks_enabled,
    transformer_start,
    transformer_end,
    single_blocks_enabled,
    single_start,
    single_end,
    image_size,
    *captions
):
    """
    Run the training process.
    """
    captions = list(captions)
    tmpdir = None

    try:
        yield "Starting training process...\n\n"
        
        yield f"[DEBUG] Input parameters:\n"
        yield f"[DEBUG] Base model: {base_model}\n"
        yield f"[DEBUG] Number of uploaded files: {len(uploaded_files)}\n"
        yield f"[DEBUG] Trigger word: {trigger_word}\n"
        yield f"[DEBUG] Epochs: {epochs}\n"
        yield f"[DEBUG] Batch size: {batch_size}\n"
        yield f"[DEBUG] LoRA rank: {lora_rank}\n"
        yield f"[DEBUG] Output directory: {output_dir}\n"
        yield f"[DEBUG] Resume checkpoint: {resume_checkpoint}\n"
        yield f"[DEBUG] Image size: {image_size}\n"
        yield f"[DEBUG] Transformer blocks enabled: {transformer_blocks_enabled}\n"
        yield f"[DEBUG] Single blocks enabled: {single_blocks_enabled}\n"
        
        target_size = int(image_size.split('x')[0])
        yield f"[DEBUG] Target size parsed: {target_size}\n"
        
        tmpdir = tempfile.mkdtemp(prefix="mflux_dreambooth_")
        yield f"[DEBUG] Created temporary directory: {tmpdir}\n"
        
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        yield f"[DEBUG] Expanded output directory: {output_dir}\n"

        config = prepare_training_config(
            base_model,
            trigger_word,
            epochs,
            batch_size,
            lora_rank,
            output_dir,
            transformer_blocks_enabled,
            transformer_start,
            transformer_end,
            single_blocks_enabled,
            single_start,
            single_end,
            image_size,
            tmpdir
        )

        yield "[DEBUG] Starting image processing\n"
        for i, (fdata, ctext) in enumerate(zip(uploaded_files, captions)):
            if not ctext:
                continue
            ext = os.path.splitext(fdata.name)[1]
            out_fn = f"img_{i}{ext}"
            out_path = os.path.join(tmpdir, out_fn)
            
            yield f"[DEBUG] Processing image {i+1}: {fdata.name} -> {out_fn}\n"
            yield f"[DEBUG] Caption: {ctext}\n"
            
            resize_image(fdata.name, out_path, target_size)
            yield f"[DEBUG] Resized and saved image to: {out_path}\n"

            config["examples"]["images"].append({
                "image": out_fn,
                "prompt": f"{ctext}"
            })

        config_path = os.path.join(tmpdir, "config.json")
        project_config_path = os.path.join(os.getcwd(), "last_training_config.json")
        
        yield f"[DEBUG] Saving config to:\n"
        yield f"[DEBUG] - Temp path: {config_path}\n"
        yield f"[DEBUG] - Project path: {project_config_path}\n"
        
        for save_path in [config_path, project_config_path]:
            with open(save_path, "w") as f:
                json.dump(config, f, indent=2)

        if not config["examples"]["images"]:
            raise ValueError("No valid image-caption pairs found in configuration")
        
        if not config["lora_layers"]:
            raise ValueError("No LoRA layers configured. Enable either transformer blocks or single transformer blocks")
        
        if not os.path.isdir(config["examples"]["path"]):
            raise ValueError(f"Images directory not found: {config['examples']['path']}")

        train_cmd = [
            "mflux-train",
            "--train-config",
            config_path,
            "--model",
            "dev" if "dev" in base_model else "schnell"
        ]
        
        if config["quantize"]:
            train_cmd.extend(["--quantize", str(config["quantize"])])
        
        if resume_checkpoint:
            train_cmd.extend(["--resume-checkpoint", resume_checkpoint])

        yield f"[DEBUG] Executing training command: {' '.join(train_cmd)}\n"
        
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1 
        )

        output_queue = queue.Queue()
        
        def enqueue_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                output_queue.put(f"[{prefix}] {line.strip()}\n")
            pipe.close()

        stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, 'STDOUT'))
        stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, 'STDERR'))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        while process.poll() is None:
            try:
                line = output_queue.get_nowait()
                yield line
            except queue.Empty:
                continue

        remaining_output = []
        while True:
            try:
                line = output_queue.get_nowait()
                remaining_output.append(line)
            except queue.Empty:
                break

        for line in remaining_output:
            yield line

        if process.returncode != 0:
            yield f"[ERROR] Training process exited with code {process.returncode}\n"
        else:
            yield "[SUCCESS] Training completed successfully\n"

    except Exception as e:
        error_details = f"Error during training: {str(e)}\n"
        error_details += f"Full traceback:\n{traceback.format_exc()}"
        yield error_details
    finally:
        if tmpdir and os.path.exists(tmpdir):
            try:
                import shutil
                shutil.rmtree(tmpdir)
                yield f"[DEBUG] Cleaned up temporary directory: {tmpdir}\n"
            except Exception as e:
                yield f"[WARNING] Failed to clean up temporary directory: {str(e)}\n"

def run_dreambooth_from_ui_no_explicit_quantize(*args, **kwargs):
    """
    Run Dreambooth training from the UI without explicit quantization.
    This is a wrapper around run_training that handles the UI-specific parameters.
    """
    progress_text = ""
    try:
        for progress in run_training(*args, **kwargs):
            progress_text += progress
            yield progress_text
    except Exception as e:
        error_text = f"Error during training: {str(e)}\n"
        error_text += f"Full traceback:\n{traceback.format_exc()}"
        yield error_text
