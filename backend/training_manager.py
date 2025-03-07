import os
import json
import subprocess
import tempfile
import traceback
import threading
import queue
from PIL import Image
from pathlib import Path
import time

# Create configs directory if it doesn't exist
configs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
os.makedirs(configs_dir, exist_ok=True)

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
    
    # Only add one type of transformer blocks based on what's enabled
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
    elif single_blocks_enabled:
        config["lora_layers"]["single_transformer_blocks"] = {
            "block_range": {
                "start": int(single_start),
                "end": min(int(single_end), 38)
            },
            "layer_types": [
                "proj_out",
                "proj_mlp",
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
    try:
        yield "Starting training process...\n\n"
        
        # Save debug config
        debug_config = {
            "base_model": base_model,
            "trigger_word": trigger_word,
            "epochs": epochs,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            "output_dir": output_dir,
            "transformer_blocks_enabled": transformer_blocks_enabled,
            "transformer_start": transformer_start, 
            "transformer_end": transformer_end,
            "single_blocks_enabled": single_blocks_enabled,
            "single_start": single_start,
            "single_end": single_end,
            "image_size": image_size,
            "captions": list(captions),
            "uploaded_files": [f.name for f in uploaded_files]
        }
        
        # Save debug config in configs directory instead of output directory
        debug_path = os.path.join(configs_dir, "dreambooth_debug_config.json")
        
        with open(debug_path, "w") as f:
            json.dump(debug_config, f, indent=2)
            
        yield f"[DEBUG] Saved debug config to: {debug_path}\n"
        
        captions = list(captions)
        tmpdir = None

        yield "Input parameters:\n"
        yield f"• Base model: {base_model}\n"
        yield f"• Number of images: {len(uploaded_files)}\n"
        yield f"• Trigger word: {trigger_word}\n"
        yield f"• Epochs: {epochs}\n"
        yield f"• Batch size: {batch_size}\n"
        yield f"• LoRA rank: {lora_rank}\n"
        yield f"• Output directory: {output_dir}\n"
        yield f"• Resume checkpoint: {resume_checkpoint}\n"
        yield f"• Image size: {image_size}\n"
        yield f"• Transformer blocks enabled: {transformer_blocks_enabled}\n"
        yield f"• Single blocks enabled: {single_blocks_enabled}\n\n"
        
        target_size = int(image_size.split('x')[0])
        yield f"Processing images to {target_size}x{target_size}...\n"
        
        tmpdir = tempfile.mkdtemp(prefix="mflux_dreambooth_")
        yield f"Created temporary directory: {tmpdir}\n"
        
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

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

        yield "\nProcessing images:\n"
        for i, (fdata, ctext) in enumerate(zip(uploaded_files, captions)):
            if not ctext:
                continue
            ext = os.path.splitext(fdata.name)[1]
            out_fn = f"img_{i}{ext}"
            out_path = os.path.join(tmpdir, out_fn)
            
            yield f"• Image {i+1}: {fdata.name}\n"
            yield f"  Caption: {ctext}\n"
            
            resize_image(fdata.name, out_path, target_size)

            config["examples"]["images"].append({
                "image": out_fn,
                "prompt": f"{ctext}"
            })

        config_path = os.path.join(tmpdir, "config.json")
        project_config_path = os.path.join(os.getcwd(), "last_training_config.json")
        
        yield "\nSaving configs:\n"
        yield f"• Temp config: {config_path}\n"
        yield f"• Project config: {project_config_path}\n"
        
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
            "python",
            os.path.join(os.path.dirname(__file__), 'custom_train.py'),
            "--train-config",
            config_path,
            "--model",
            "dev" if "dev" in base_model else "schnell"
        ]
        
        if config["quantize"]:
            train_cmd.extend(["--quantize", str(config["quantize"])])
        
        # Add env variable to skip huggingface hub authentication by setting the token to be empty
        os.environ["HF_TOKEN"] = ""
        os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
            
        # Voeg low-ram optie toe indien nodig
        train_cmd.append("--low-ram")
        
        # Gebruik lokaal model pad indien beschikbaar
        if "dev" in base_model:
            model_name = "AITRADER/MFLUXUI.1-dev"
        else:
            model_name = "AITRADER/MFLUXUI.1-schnell"
            
        # Zoek in de Hugging Face cache naar een geschikt model
        local_model_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots")
        
        if os.path.exists(local_model_path):
            # Get the most recent snapshot directory
            snapshot_dirs = [d for d in os.listdir(local_model_path) if os.path.isdir(os.path.join(local_model_path, d))]
            if snapshot_dirs:
                # Sort by name to get the most recent snapshot (assuming hash-based naming)
                snapshot_dir = sorted(snapshot_dirs)[-1]
                full_path = os.path.join(local_model_path, snapshot_dir)
                train_cmd.extend(["--path", full_path])
                yield f"Using local model path: {full_path}\n"
            else:
                yield f"No snapshot directories found in {local_model_path}\n"
        else:
            yield f"Local model path not found: {local_model_path}\n"
            yield "Using preloaded models from environment\n"
        
        if resume_checkpoint:
            train_cmd.extend(["--resume-checkpoint", resume_checkpoint])

        yield f"\nStarting training with command:\n{' '.join(train_cmd)}\n\n"
        
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

        # Buffer voor de output
        output_buffer = ""
        last_yield_time = time.time()
        
        while process.poll() is None:
            try:
                line = output_queue.get_nowait()
                output_buffer += line
                
                # Yield elke seconde of als de buffer groot genoeg is
                current_time = time.time()
                if current_time - last_yield_time > 1.0 or len(output_buffer) > 500:
                    yield output_buffer
                    output_buffer = ""
                    last_yield_time = current_time
                    
            except queue.Empty:
                if output_buffer:  # Als er nog iets in de buffer zit
                    yield output_buffer
                    output_buffer = ""
                time.sleep(0.1)  # Voorkom CPU spinning
                continue

        # Verzamel overgebleven output
        remaining_output = []
        while True:
            try:
                line = output_queue.get_nowait()
                remaining_output.append(line)
            except queue.Empty:
                break

        if remaining_output:
            yield "".join(remaining_output)

        if process.returncode != 0:
            yield f"\n❌ Training process failed with code {process.returncode}\n"
        else:
            yield "\n✅ Training completed successfully!\n"

    except Exception as e:
        error_details = f"Error during training: {str(e)}\n"
        error_details += f"Full traceback:\n{traceback.format_exc()}"
        yield error_details
    finally:
        if tmpdir and os.path.exists(tmpdir):
            try:
                import shutil
                shutil.rmtree(tmpdir)
                yield f"\nCleaned up temporary directory: {tmpdir}\n"
            except Exception as e:
                yield f"\nWarning: Failed to clean up temporary directory: {str(e)}\n"

def run_dreambooth_from_ui_no_explicit_quantize(*args, **kwargs):
    """
    Run Dreambooth training from the UI without explicit quantization.
    This is a wrapper around run_training that handles the UI-specific parameters.
    """
    progress_text = ""
    try:
        # Save debug config first to ensure it's always saved
        debug_config = {
            "base_model": args[0],
            "trigger_word": args[2],
            "epochs": args[3],
            "batch_size": args[4],
            "lora_rank": args[5],
            "output_dir": args[6],
            "transformer_blocks_enabled": args[9],
            "transformer_start": args[10], 
            "transformer_end": args[11],
            "single_blocks_enabled": args[12],
            "single_start": args[13],
            "single_end": args[14],
            "image_size": args[15],
            "captions": list(args[16:]),
            "uploaded_files": [f.name for f in args[1]]
        }
        
        debug_path = os.path.join(configs_dir, "dreambooth_debug_config.json")
        os.makedirs(configs_dir, exist_ok=True)
        
        with open(debug_path, "w") as f:
            json.dump(debug_config, f, indent=2)
        
        progress_text += f"[DEBUG] Saved debug config to: {debug_path}\n"
        
        # Now run the training process
        for progress in run_training(*args, **kwargs):
            progress_text += progress
            yield progress_text
    except Exception as e:
        error_text = f"Error during training: {str(e)}\n"
        error_text += f"Full traceback:\n{traceback.format_exc()}"
        yield error_text
