from safetensors import safe_open

file_path = '/Users/charafchnioune/Desktop/code/MFLUX-WEBUI/lora/optimus-optimus.safetensors'

with safe_open(file_path, framework="pt") as f:
    metadata = f.metadata()
    print(metadata)
