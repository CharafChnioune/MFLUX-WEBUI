#!/usr/bin/env python3
"""
Batch convert all LoRAs to MFLUX format.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from lora_converter import convert_lora_keys

def main():
    lora_dir = Path(__file__).parent / "lora"
    
    # Find all non-mflux LoRAs
    all_loras = list(lora_dir.glob("*.safetensors"))
    to_convert = [f for f in all_loras if not f.stem.endswith("_mflux")]
    
    print(f"📁 Found {len(all_loras)} total LoRAs", flush=True)
    print(f"🔄 Need to convert: {len(to_convert)}", flush=True)
    
    success = 0
    skip = 0
    fail = 0
    
    for idx, lora_path in enumerate(to_convert, 1):
        output_path = lora_path.parent / f"{lora_path.stem}_mflux.safetensors"
        
        # Skip if already converted
        if output_path.exists():
            print(f"[{idx}/{len(to_convert)}] ⏭️  {lora_path.name} (exists)", flush=True)
            skip += 1
            continue
        
        try:
            print(f"[{idx}/{len(to_convert)}] 🔄 {lora_path.name[:50]}...", flush=True)
            convert_lora_keys(str(lora_path), str(output_path))
            print(f"   ✅ Converted", flush=True)
            success += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}", flush=True)
            fail += 1
    
    print("\n" + "="*50, flush=True)
    print(f"✅ Converted: {success}", flush=True)
    print(f"⏭️  Skipped: {skip}", flush=True)
    print(f"❌ Failed: {fail}", flush=True)
    print("="*50, flush=True)

if __name__ == "__main__":
    main()
