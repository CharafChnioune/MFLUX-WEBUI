#!/usr/bin/env python3
"""
CivitAI LoRA downloader with pagination - downloads ALL FLUX LoRAs.
"""
import os
import sys
import json
import requests
from pathlib import Path
import argparse
import time

def load_config():
    """Load API key from config"""
    config_path = Path(__file__).parent / "backend" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

def download_model(model_id, output_dir, api_key):
    """Download a single model by ID - only if it's a LoRA"""
    api_url = f"https://civitai.com/api/v1/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        model_data = response.json()
        
        model_name = model_data.get("name", f"model_{model_id}")
        model_type = model_data.get("type", "Unknown")
        
        # SKIP if not a LoRA
        if model_type.upper() != "LORA":
            print(f"⏭️  Skipping {model_name} (type: {model_type})", flush=True)
            return "skip"
            
        versions = model_data.get("modelVersions", [])
        
        if not versions:
            print(f"⚠️  No versions found for {model_name}", flush=True)
            return False
            
        version = versions[0]
        files = version.get("files", [])
        
        downloaded = False
        for file_info in files:
            filename = file_info.get("name")
            if not filename or not filename.endswith((".safetensors", ".ckpt", ".pt")):
                continue
                
            download_url = file_info.get("downloadUrl")
            if not download_url:
                continue
                
            # Add API key to URL
            separator = "&" if "?" in download_url else "?"
            download_url = f"{download_url}{separator}token={api_key}"
            
            filepath = output_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                print(f"⏭️  {filename} (exists)", flush=True)
                return "skip"
                
            print(f"⬇️  {model_name[:40]}...", flush=True)
            
            # Download with streaming
            with requests.get(download_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                
                with open(filepath, "wb") as f:
                    downloaded_bytes = 0
                    for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
            
            size_mb = total_size // (1024*1024)
            print(f"   ✅ {filename} ({size_mb}MB)", flush=True)
            downloaded = True
            
        return downloaded
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error {model_id}: {e}", flush=True)
        return False
    except Exception as e:
        print(f"❌ Unexpected {model_id}: {e}", flush=True)
        return False

def search_all_flux_loras(api_key, max_models=None):
    """Search for ALL FLUX LoRAs using pagination"""
    all_model_ids = []
    page = 1
    
    while True:
        search_url = "https://civitai.com/api/v1/models"
        params = {
            "types": "LORA",
            "sort": "Highest Rated",
            "limit": 100,  # Max per page
            "page": page,
            "nsfw": "true",
            "tag": "flux"
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            print(f"📄 Fetching page {page}...", flush=True)
            response = requests.get(search_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            models = data.get("items", [])
            if not models:
                break  # No more results
            
            page_ids = [m.get("id") for m in models if m.get("id")]
            all_model_ids.extend(page_ids)
            
            print(f"   Found {len(page_ids)} models (total: {len(all_model_ids)})", flush=True)
            
            # Check if we have more pages
            metadata = data.get("metadata", {})
            current_page = metadata.get("currentPage", page)
            total_pages = metadata.get("totalPages", 1)
            
            if current_page >= total_pages:
                break  # Last page
            
            if max_models and len(all_model_ids) >= max_models:
                all_model_ids = all_model_ids[:max_models]
                break
            
            page += 1
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"❌ Error fetching page {page}: {e}", flush=True)
            break
    
    print(f"\n🔍 Total FLUX LoRAs found: {len(all_model_ids)}", flush=True)
    return all_model_ids

def download_all_models(output_dir, api_key, limit=None):
    """Download all LoRAs from CivitAI"""
    print("🔍 Searching for ALL FLUX LoRAs...", flush=True)
    model_ids = search_all_flux_loras(api_key, max_models=limit)
    
    if not model_ids:
        print("❌ No LoRAs found", flush=True)
        return
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    print(f"\n⬇️  Starting downloads...\n", flush=True)
    
    for idx, model_id in enumerate(model_ids, 1):
        print(f"[{idx}/{len(model_ids)}] ", end="", flush=True)
        
        result = download_model(model_id, output_dir, api_key)
        if result is True:
            success_count += 1
        elif result == "skip":
            skip_count += 1
        else:
            fail_count += 1
        
        # Progress summary every 20 models
        if idx % 20 == 0:
            print(f"\n📊 Progress: {success_count} new | {skip_count} skipped | {fail_count} failed\n", flush=True)
    
    # Final summary
    print("\n" + "="*50, flush=True)
    print(f"✅ Downloaded: {success_count}", flush=True)
    print(f"⏭️  Skipped: {skip_count}", flush=True)
    print(f"❌ Failed: {fail_count}", flush=True)
    print("="*50, flush=True)

def main():
    parser = argparse.ArgumentParser(description="CivitAI LoRA Downloader")
    parser.add_argument("--download-all", action="store_true", help="Download all LoRAs")
    parser.add_argument("--model-id", type=int, help="Download specific model by ID")
    parser.add_argument("--output-dir", type=str, default="lora", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of models")
    
    args = parser.parse_args()
    
    # Load API key
    config = load_config()
    api_key = config.get("civitai_api_key", "")
    
    if not api_key:
        print("❌ No CivitAI API key found in config.json", flush=True)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output: {output_dir.absolute()}", flush=True)
    print(f"🔑 API key loaded\n", flush=True)
    
    if args.download_all:
        download_all_models(output_dir, api_key, args.limit)
    elif args.model_id:
        download_model(args.model_id, output_dir, api_key)
    else:
        print("❌ Specify --download-all or --model-id", flush=True)
        sys.exit(1)
    
    print("\n✨ Done!", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
