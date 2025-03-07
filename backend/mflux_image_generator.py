import os
import time
import json
import subprocess
from pathlib import Path
from crewai.tools import BaseTool
from backend.flux_manager import get_or_create_flux

# Globale variabelen om UI-parameters op te slaan
GLOBAL_UI_MODEL = None
GLOBAL_UI_STEPS = None
GLOBAL_UI_LORA_FILES = None
GLOBAL_UI_LORA_SCALES = None

def set_global_ui_parameters(model, steps, lora_files, lora_scales):
    """
    Sla de UI-parameters globaal op zodat ze gebruikt kunnen worden door de CrewAI-tools
    zonder dat ze via de agent worden doorgegeven.
    """
    global GLOBAL_UI_MODEL, GLOBAL_UI_STEPS, GLOBAL_UI_LORA_FILES, GLOBAL_UI_LORA_SCALES
    GLOBAL_UI_MODEL = model
    GLOBAL_UI_STEPS = steps
    GLOBAL_UI_LORA_FILES = lora_files
    GLOBAL_UI_LORA_SCALES = lora_scales
    print(f"Global UI parameters set: model={model}, steps={steps}, lora_files={lora_files}, lora_scales={lora_scales}")

class SimplifiedImageGeneratorTool(BaseTool):
    name: str = "MFLUX Image Generator"
    description: str = "Generates images using the MFLUX image generator with a prompt (other parameters are set by the UI)"
    
    def _run(self, prompt: str) -> str:
        """
        Simplified image generator tool that only accepts a prompt.
        All other parameters (model, steps, lora_files, lora_scales) are taken from global UI settings.
        
        Args:
            prompt: The text description for the image to generate
        
        Returns:
            Path to the generated image(s)
        """
        global GLOBAL_UI_MODEL, GLOBAL_UI_STEPS, GLOBAL_UI_LORA_FILES, GLOBAL_UI_LORA_SCALES
        
        # Gebruik de globale UI-parameters
        model = GLOBAL_UI_MODEL
        steps = GLOBAL_UI_STEPS
        lora_files = GLOBAL_UI_LORA_FILES
        lora_scales = GLOBAL_UI_LORA_SCALES
        
        print(f"SimplifiedImageGeneratorTool using UI parameters: model={model}, steps={steps}, lora_files={lora_files}, lora_scales={lora_scales}")
        
        # Standaard waarden indien niet beschikbaar
        if model is None:
            from backend.model_manager import get_updated_models
            available_models = get_updated_models()
            model = available_models[0] if available_models else "schnell-4-bit"
        
        if steps is None:
            steps = 4 if "schnell" in model else 20
        
        # Process prompt
        if isinstance(prompt, dict) and "description" in prompt:
            prompt = prompt["description"]
        elif isinstance(prompt, dict) and "prompt" in prompt:
            prompt = prompt["prompt"]
        
        print(f"Generating image with prompt: {prompt}")
        print(f"Using model: {model}, steps: {steps}")
        
        # Verwerk LoRA bestanden
        lora_files_processed = None
        lora_scales_processed = None
        
        if lora_files and len(lora_files) > 0:
            # Zorg dat lora_files een lijst is
            if not isinstance(lora_files, list):
                if isinstance(lora_files, str) and ',' in lora_files:
                    lora_files = [f.strip() for f in lora_files.split(',')]
                else:
                    lora_files = [lora_files]
            
            print(f"Processing LoRA files: {lora_files}")
            
            # Verwerk lora_scales
            lora_scales_list = []
            if lora_scales is None:
                lora_scales_list = [1.0] * len(lora_files)
            elif isinstance(lora_scales, (int, float)):
                lora_scales_list = [float(lora_scales)] * len(lora_files)
            elif isinstance(lora_scales, list):
                # Zorg dat alle schalen float zijn
                lora_scales_list = [float(scale) if isinstance(scale, (int, float, str)) and str(scale).replace('.', '', 1).isdigit() else 1.0 for scale in lora_scales]
                
                # Correct de lengte
                if len(lora_scales_list) < len(lora_files):
                    lora_scales_list.extend([1.0] * (len(lora_files) - len(lora_scales_list)))
                elif len(lora_scales_list) > len(lora_files):
                    lora_scales_list = lora_scales_list[:len(lora_files)]
            else:
                lora_scales_list = [1.0] * len(lora_files)
            
            # Log LoRA info
            lora_info = ", ".join([f"{lora} (scale: {scale})" for lora, scale in zip(lora_files, lora_scales_list)])
            print(f"Using LoRA files: {lora_info}")
            
            # Process LoRA files
            from backend.lora_manager import process_lora_files
            lora_files_processed = process_lora_files(lora_files)
            
            if lora_files_processed:
                # Gebruik dezelfde schaal-verwerking als hierboven
                lora_scales_processed = []
                for i, _ in enumerate(lora_files):
                    if i < len(lora_scales_list):
                        scale = lora_scales_list[i]
                        # Begrens tussen 0.1 en 1.5
                        scale = max(0.1, min(1.5, scale))
                        lora_scales_processed.append(scale)
                    else:
                        lora_scales_processed.append(1.0)
                
                print(f"Processed LoRA files: {lora_files_processed}")
                print(f"Processed LoRA scales: {lora_scales_processed}")
            
        # Gebruik standaard steps als niet opgegeven
        if steps is None:
            steps = 4 if "schnell" in model else 20
        elif isinstance(steps, str):
            steps = int(steps)
        
        # Genereer afbeeldingen
        from backend.flux_manager import get_or_create_flux, generate_image_batch
        
        # Maak Flux instantie
        flux = get_or_create_flux(model, None, None, lora_files_processed, lora_scales_processed)
        
        # Standaard instellingen
        guidance = 1.0 if "schnell" in model else 7.5
        width, height = 576, 1024
        num_images = 1  # Altijd één afbeelding genereren per keer
        
        # Genereer de afbeeldingen
        try:
            images, filenames, seeds = generate_image_batch(
                flux=flux,
                prompt=prompt,
                seed=None,
                steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                num_images=num_images
            )
            
            # Wacht even om zeker te zijn
            time.sleep(2)
            
            # Verwerk resultaten
            full_paths = []
            for filename in filenames:
                full_path = os.path.abspath(filename)
                full_paths.append(full_path)
                print(f"Generated image at: {full_path}")
            
            if len(full_paths) == 1:
                return full_paths[0]
            
            return "\n".join(full_paths)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in SimplifiedImageGeneratorTool: {str(e)}\n{error_details}")
            return f"Image generation failed: {str(e)}"


class MfluxImageGeneratorTool(BaseTool):
    name: str = "MFLUX Image Generator"
    description: str = "Generates images using the MFLUX image generator with specific prompts"
    
    def _run(self, prompt: str, model: str = None, num_images: int = 1, steps: int = None, lora_files = None, lora_scales = None) -> str:
        """
        Generates an image using the MFLUX image generator and returns the output file path.
        
        Args:
            prompt: The text description for the image to generate
            model: The model to use (van gebruiker UI)
            num_images: Number of images to generate
            steps: Number of inference steps (van gebruiker UI)
            lora_files: List of LoRA files to use (van gebruiker UI)
            lora_scales: List of scale factors for each LoRA effect (van gebruiker UI)
        
        Returns:
            Path to the generated image(s)
        """
        try:
            # Fix voor verschillende prompt formaten van CrewAI agents
            # Soms stuurt de agent een dict met description field in plaats van een string
            if isinstance(prompt, dict) and "description" in prompt:
                prompt = prompt["description"]
            elif isinstance(prompt, str):
                # Controleer of het een JSON string is dat ontleed kan worden
                try:
                    prompt_data = json.loads(prompt)
                    if isinstance(prompt_data, dict):
                        if "prompt" in prompt_data:
                            # {"prompt": "text"} format
                            if isinstance(prompt_data["prompt"], str):
                                prompt = prompt_data["prompt"]
                            # {"prompt": {"description": "text"}} format
                            elif isinstance(prompt_data["prompt"], dict) and "description" in prompt_data["prompt"]:
                                prompt = prompt_data["prompt"]["description"]
                except (json.JSONDecodeError, TypeError):
                    # Het was geen geldige JSON string, gebruik het originele prompt
                    pass
            
            # Log het uiteindelijke prompt voor debugging
            print(f"Processed prompt for image generation: {prompt[:100]}...")

            # Haal de globale instellingen op uit de UI configuratie
            from backend.model_manager import get_updated_models

            # Controleer parameters en gebruik defaults alleen als ze helemaal niet meegegeven zijn
            if model is None or model == "":
                # Fallback naar een standaard model alleen als niets is gespecificeerd
                available_models = get_updated_models()
                model = available_models[0] if available_models else "schnell-4-bit"
            
            # Process model uit prompt format
            if isinstance(model, str):
                try:
                    model_data = json.loads(model)
                    if isinstance(model_data, dict) and "model" in model_data:
                        model = model_data["model"]
                except (json.JSONDecodeError, TypeError):
                    # Geen geldige json, gebruik originele
                    pass
            
            # Process num_images uit prompt format
            if isinstance(num_images, str):
                try:
                    num_images_data = json.loads(num_images)
                    if isinstance(num_images_data, dict) and "num_images" in num_images_data:
                        num_images = num_images_data["num_images"]
                except (json.JSONDecodeError, TypeError):
                    # Geen geldige json, gebruik originele
                    try:
                        num_images = int(num_images)
                    except (ValueError, TypeError):
                        num_images = 1
            
            # Process steps uit prompt format
            if isinstance(steps, str):
                try:
                    steps_data = json.loads(steps)
                    if isinstance(steps_data, dict) and "steps" in steps_data:
                        steps = steps_data["steps"]
                except (json.JSONDecodeError, TypeError):
                    # Geen geldige json, gebruik originele
                    try:
                        steps = int(steps)
                    except (ValueError, TypeError):
                        steps = None
            
            # Process lora_files uit prompt format
            if isinstance(lora_files, str):
                try:
                    lora_data = json.loads(lora_files)
                    if isinstance(lora_data, dict) and "lora_files" in lora_data:
                        lora_files = lora_data["lora_files"]
                except (json.JSONDecodeError, TypeError):
                    # Geen geldige json, proberen als string lijst te parsen
                    pass
            
            # Process lora_scales uit prompt format
            if isinstance(lora_scales, str):
                try:
                    lora_scales_data = json.loads(lora_scales)
                    if isinstance(lora_scales_data, dict) and "lora_scales" in lora_scales_data:
                        lora_scales = lora_scales_data["lora_scales"]
                except (json.JSONDecodeError, TypeError):
                    # Geen geldige json, proberen als string lijst te parsen
                    pass

            # Debug informatie tonen over de ontvangen LoRA instellingen
            print(f"LoRA files received: {lora_files}")
            print(f"LoRA scales received: {lora_scales}")
            print(f"Generating image with prompt: {prompt}")
            print(f"Using model: {model}, steps: {steps}")
            
            # Controleer en verwerk lora_files om te zorgen dat we een geldige lijst hebben
            lora_files_processed = None
            lora_scales_processed = None
            
            if lora_files and len(lora_files) > 0:
                # Zorg dat lora_files een lijst is
                if not isinstance(lora_files, list):
                    if isinstance(lora_files, str) and ',' in lora_files:
                        # Mogelijk een komma-gescheiden string
                        lora_files = [f.strip() for f in lora_files.split(',')]
                    else:
                        # Een enkele waarde
                        lora_files = [lora_files]
                        
                print(f"Normalized LoRA files to list: {lora_files}")
                
                lora_scales_list = []
                
                # Controleer lora_scales parameter
                if lora_scales is None:
                    lora_scales_list = [1.0] * len(lora_files)
                elif isinstance(lora_scales, (int, float)):
                    lora_scales_list = [float(lora_scales)] * len(lora_files)
                elif isinstance(lora_scales, list):
                    # Converteer alle waarden naar float
                    lora_scales_list = []
                    for scale in lora_scales:
                        if isinstance(scale, (int, float)):
                            lora_scales_list.append(float(scale))
                        elif isinstance(scale, str):
                            try:
                                lora_scales_list.append(float(scale))
                            except (ValueError, TypeError):
                                lora_scales_list.append(1.0)
                        else:
                            lora_scales_list.append(1.0)
                    
                    # Zorg dat de lijst de juiste lengte heeft
                    if len(lora_scales_list) < len(lora_files):
                        lora_scales_list.extend([1.0] * (len(lora_files) - len(lora_scales_list)))
                    elif len(lora_scales_list) > len(lora_files):
                        lora_scales_list = lora_scales_list[:len(lora_files)]
                else:
                    lora_scales_list = [1.0] * len(lora_files)
                
                # Log lora informatie
                lora_info = ""
                for i, lora in enumerate(lora_files):
                    lora_name = os.path.basename(lora) if os.path.exists(lora) else lora
                    scale = lora_scales_list[i] if i < len(lora_scales_list) else 1.0
                    lora_info += f"{lora_name} (scale: {scale}), "
                
                print(f"Using LoRA files: {lora_info}")
                
                # Gebruik process_lora_files om de werkelijke bestandspaden te krijgen
                from backend.lora_manager import process_lora_files, get_lora_choices, get_available_lora_files
                
                # Debug informatie weergeven
                available_choices = get_lora_choices()
                print(f"Available LoRA choices: {available_choices}")
                
                # Verwerk de LoRA bestanden
                lora_files_processed = process_lora_files(lora_files)
                print(f"Processed LoRA files: {lora_files_processed}")
                
                if lora_files_processed:
                    lora_scales_processed = []
                    for i, lora in enumerate(lora_files):
                        if i < len(lora_scales_list):
                            scale = lora_scales_list[i]
                            # Begrens scale waarden tussen 0.1 en 1.5
                            if scale < 0.1:
                                scale = 0.1
                            elif scale > 1.5:
                                scale = 1.5
                            lora_scales_processed.append(scale)
                        else:
                            lora_scales_processed.append(1.0)
                            
                    print(f"Final LoRA scales: {lora_scales_processed}")
            
            # Stel default steps in op basis van het model ALLEEN als er geen steps zijn meegegeven
            if steps is None:
                steps = 4 if "schnell" in model else 20
            
            # Gebruik de bestaande MFLUX backend om afbeeldingen te genereren
            from backend.flux_manager import get_or_create_flux, generate_image_batch
            
            # Maak de Flux-instantie
            flux = get_or_create_flux(model, None, None, lora_files_processed, lora_scales_processed)
            
            # Zet de waarden om naar de juiste types
            if isinstance(steps, str):
                steps = int(steps)
            
            guidance = 1.0 if "schnell" in model else 7.5
            
            # Standaard formaat voor image (Portrait 576x1024)
            width, height = 576, 1024
            
            # Genereer afbeeldingen
            images, filenames, seeds = generate_image_batch(
                flux=flux,
                prompt=prompt,
                seed=None,
                steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                num_images=int(num_images) if isinstance(num_images, str) else num_images
            )
            
            # We wachten even om zeker te zijn dat het beeld gegenereerd is
            time.sleep(2)
            
            # Log het absolute pad van de gegenereerde afbeeldingen
            full_paths = []
            for filename in filenames:
                full_path = os.path.abspath(filename)
                full_paths.append(full_path)
                print(f"Generated image at: {full_path}")
            
            # Controleer of de output_path een lijst is (meerdere afbeeldingen) of een enkele string
            if len(full_paths) == 1:
                return full_paths[0]
            
            return "\n".join(full_paths)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in MfluxImageGeneratorTool: {str(e)}\n{error_details}")
            return f"Image generation failed: {str(e)}"

def generate_image_tool():
    """Returns an initialized MFLUX image generator tool"""
    return MfluxImageGeneratorTool() 

def generate_simplified_image_tool():
    """Returns an initialized Simplified MFLUX image generator tool that only requires a prompt"""
    return SimplifiedImageGeneratorTool() 