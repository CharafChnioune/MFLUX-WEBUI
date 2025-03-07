import gradio as gr
import os
from backend.crew import run_photo_crew, create_photo_crew
from backend.simplified_crew import run_simplified_crew
from backend.direct_generator import generate_direct_image
from backend.lora_manager import get_lora_choices, update_lora_scales
from backend.model_manager import get_updated_models
from backend.llm_manager import LLM_PROVIDERS, get_available_models, fetch_models_from_api

# Maximum number of LoRA models that can be used simultaneously
MAX_LORAS = 5

def create_photo_crew_tab():
    """Create a Gradio tab for the AI PhotoCrew."""
    
    with gr.Blocks() as photo_crew_tab:
        gr.Markdown("""
        # 📸 AI PhotoCrew
        
        A fully automated team of AI agents working together to create beautiful images.
        
        ## How it works:
        1. **Location Scout** finds the perfect location for the photoshoot
        2. **Stylist** defines the color schemes and style
        3. **Prompt Engineer** optimizes the prompt for the best result
        4. **Photographer** generates the images with MFLUX
        5. **Art Director** analyzes the images and gives feedback
        6. **Iterative process** where each new version is improved based on feedback
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # LLM settings - Moved out of advanced options to always be visible
                gr.Markdown("### CrewAI LLM Settings")
                with gr.Row():
                    with gr.Column(scale=3):
                        # User Interface for LLM Provider settings
                        llm_provider = gr.Dropdown(
                            choices=["None"] + LLM_PROVIDERS,
                            label="LLM Provider",
                            info="Select a provider for the LLM used by the agents",
                            value="None"
                        )
                        
                        # Default options for models (will be updated on provider selection)
                        default_models = ["None LLM selected"]  
                        llm_model = gr.Dropdown(
                            choices=default_models,
                            value=default_models[0],
                            label="LLM Model",
                            info="Select a model (click 'Refresh Models' after provider selection)"
                        )
                        
                        llm_api_key = gr.Textbox(
                            label="API Key (optional)",
                            placeholder="sk-...",
                            info="API key for the selected provider (not needed for local models)",
                            visible=True
                        )
                        
                        llm_base_url = gr.Textbox(
                            label="API Base URL (optional)",
                            placeholder="http://localhost:1234/v1",
                            info="Base URL for API calls (only needed for non-standard endpoints)",
                            visible=True
                        )
                        
                        refresh_models_button = gr.Button("Refresh Models")
                
                # Extra information for LM Studio and Ollama
                lmstudio_info = gr.Markdown(
                    """
                    ### LM Studio Setup Instructions:
                    1. Start LM Studio and load a model
                    2. Go to "Server" tab and click "Start Server"
                    3. Model name fill in as shown in LM Studio (e.g. "qwen2.5-vl-72b-instruct")
                    4. API Base URL is standard: `http://localhost:1234/v1`
                    5. API Key can be set to 'dummy-key'
                    
                    If model retrieval doesn't work, you can manually specify the model name.
                    """,
                    visible=False
                )
                
                ollama_info = gr.Markdown(
                    """
                    ### Ollama Setup Instructions:
                    1. Install Ollama from [ollama.ai](https://ollama.ai)
                    2. Open a terminal and download a model: `ollama pull llama3` or `ollama pull mistral`
                    3. Ensure Ollama is running: `ollama serve` (in a separate terminal) 
                    4. Click the 🔄 button to get available models
                    5. API Base URL is standard: `http://localhost:11434`
                    
                    Tip: Model names can be: `llama3`, `mistral`, `phi3`, etc.
                    """,
                    visible=False
                )
                
                # Main input fields
                topic_input = gr.Textbox(
                    label="Theme / Concept",
                    placeholder="Cyberpunk city at night",
                    info="Describe the subject or theme for your photoshoot"
                )
                
                with gr.Row():
                    num_images_input = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Number of images",
                        info="How many images do you want to generate? (max 5)"
                    )
                    
                    advanced_toggle = gr.Checkbox(
                        label="Advanced options",
                        value=False
                    )
                
                # Interface to show agent conversations
                with gr.Accordion("Agent Conversations", open=False):
                    agent_conversations = gr.Markdown(
                        value="Agent conversations will appear here...",
                        label="Agent Conversations"
                    )
                
                with gr.Row(visible=False) as advanced_options:
                    with gr.Column(scale=1):
                        # MFLUX settings
                        gr.Markdown("### MFLUX Settings")
                        model_dropdown = gr.Dropdown(
                            choices=get_updated_models(),
                            label="MFLUX Model",
                            value="schnell-4-bit",
                            info="The model used for image generation"
                        )
                        
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=4,
                            step=1,
                            label="Number of steps",
                            info="More steps = better quality but longer generation time"
                        )
                        
                        # LoRA settings
                        gr.Markdown("### LoRA Settings")
                        lora_files = gr.Dropdown(
                            choices=get_lora_choices(),
                            label="LoRA models",
                            info="Select LoRAs to apply to your images",
                            multiselect=True,
                            value=[]
                        )
                        
                        # Container for LoRA scales
                        with gr.Group() as lora_scales_group:
                            lora_scales = []
                            for i in range(MAX_LORAS):
                                scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.05,
                                    label=f"LoRA {i+1} scale",
                                    info="The strength of the LoRA effect (1.0 = standard)",
                                    visible=False
                                )
                                lora_scales.append(scale)
            
            with gr.Column(scale=1):
                start_button = gr.Button("Start AI Photoshoot", variant="primary", size="lg")
                clear_button = gr.Button("Clear Results", variant="secondary")
                refresh_button = gr.Button("Refresh LoRA List", variant="secondary")
        
        crew_output = gr.Markdown(
            label="AI Crew Output",
            value="Start a new photoshoot to see results...",
        )
        
        with gr.Row() as gallery_row:
            gallery = gr.Gallery(
                label="Generated images",
                show_label=True,
                elem_id="gallery",
                columns=2
            )
        
        # Function to show/hide advanced options
        def toggle_advanced(show):
            return gr.update(visible=show)
            
        advanced_toggle.change(
            fn=toggle_advanced,
            inputs=advanced_toggle,
            outputs=advanced_options
        )
        
        # Function to refresh LoRA list
        def refresh_loras():
            return gr.update(choices=get_lora_choices())
            
        refresh_button.click(
            fn=refresh_loras,
            inputs=[],
            outputs=lora_files
        )
        
        # Function to make LoRA scales visible
        def update_lora_scale_visibility(selected_loras):
            updates = []
            for i in range(MAX_LORAS):
                if i < len(selected_loras) and selected_loras[i]:
                    lora_name = os.path.basename(selected_loras[i]) if os.path.exists(selected_loras[i]) else selected_loras[i]
                    updates.append(gr.update(
                        visible=True,
                        label=f"LoRA scale: {lora_name}"
                    ))
                else:
                    updates.append(gr.update(visible=False))
            return updates
            
        lora_files.change(
            fn=update_lora_scale_visibility,
            inputs=lora_files,
            outputs=lora_scales
        )
        
        # Function to update models and info panels
        def update_models_and_info(provider, api_key=None, base_url=None):
            # Determine which info panels should be visible
            lmstudio_visible = (provider == "lmstudio")
            ollama_visible = (provider == "ollama")
            
            if provider == "None":
                # If no provider selected, show an empty list
                default_models = ["None LLM selected"]
                return gr.update(choices=default_models, value=default_models[0]), gr.update(visible=lmstudio_visible), gr.update(visible=ollama_visible)
            
            try:
                # Get models
                models = get_available_models(provider)
                
                # Ensure we always have a non-empty list
                if not models:
                    models = ["default-model"] 
                    print(f"No models found for {provider}, using 'default-model' as fallback")
                
                return gr.update(choices=models, value=models[0]), gr.update(visible=lmstudio_visible), gr.update(visible=ollama_visible)
            except Exception as e:
                print(f"Error in update_models_and_info: {e}")
                # In case of an error, ensure a fallback option
                default_models = ["default-model"]
                return gr.update(choices=default_models, value=default_models[0]), gr.update(visible=lmstudio_visible), gr.update(visible=ollama_visible)
        
        # Function to manually refresh models
        def refresh_models(provider, api_key=None, base_url=None):
            if provider == "None":
                return gr.update(choices=[""], value="")
            
            print(f"Refreshing models for provider: {provider}")
            models = []
            
            try:
                if provider == "lmstudio":
                    # For LM Studio, try to connect first
                    import requests
                    try:
                        lmstudio_url = base_url or "http://localhost:1234/v1"
                        print(f"Connecting to LM Studio at {lmstudio_url}...")
                        response = requests.get(f"{lmstudio_url}/models", timeout=3)
                        if response.status_code == 200:
                            models_data = response.json().get("data", [])
                            models = [model["id"] for model in models_data]
                            print(f"Found LM Studio models: {models}")
                    except Exception as e:
                        print(f"LM Studio server not reachable: {e}")
                    
                    # If no models found or error occurred, use standard options
                    if not models:
                        models = ["mistral", "llama-3", "qwen2", "phi-3", "command-r", "gemma2", "local-model"]
                        print(f"No models found, using standard options: {models}")
                
                elif provider == "ollama":
                    # For Ollama, try to connect
                    import requests
                    try:
                        ollama_url = base_url or "http://localhost:11434"
                        print(f"Connecting to Ollama at {ollama_url}...")
                        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
                        if response.status_code == 200:
                            models = [model["name"] for model in response.json().get("models", [])]
                            print(f"Found Ollama models: {models}")
                    except Exception as e:
                        print(f"Ollama server not reachable: {e}")
                    
                    # If no models found or error occurred, use standard options
                    if not models:
                        models = ["llama3", "llama3:8b", "mistral", "phi3", "mixtral:8x7b", "gemma2", "codellama:7b"]
                        print(f"No models found, using standard options: {models}")
                
                # For other providers, try to dynamically get models
                else:
                    models = fetch_models_from_api(provider, api_key, base_url)
                    print(f"Retrieved models for {provider}: {models}")
                    if not models:  # Fallback if no models found
                        models = get_available_models(provider)
                        print(f"Fallback to standard models: {models}")
                
                # Ensure we always have a non-empty list
                if not models:
                    models = ["default-model"]
                    print("No models found, using 'default-model' as fallback")
                
                return gr.update(choices=models, value=models[0])
                
            except Exception as e:
                import traceback
                print(f"Error in retrieving models: {e}")
                traceback.print_exc()
                
                # In case of an error, use fallback
                models = get_available_models(provider)
                if not models:
                    models = ["default-model"]
                    
                print(f"Fallback models after error: {models}")
                return gr.update(choices=models, value=models[0])
                
        # Event handlers
        llm_provider.change(
            fn=update_models_and_info,
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=[llm_model, lmstudio_info, ollama_info]
        )
        
        refresh_models_button.click(
            fn=refresh_models,
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model
        )
        
        # Function to process results
        def run_crew(topic, num_images, model, steps, lora_files, llm_provider, llm_model, llm_api_key, llm_base_url, *lora_scales_args):
            if not topic:
                yield "Enter a theme to start.", [], "Agent conversations will appear here..."
                return
            
            try:
                # Filter lora_scales only for the selected LoRAs
                lora_scales = []
                if lora_files:
                    lora_scales = list(lora_scales_args[:len(lora_files)])
                
                # Process LLM settings
                final_llm_provider = None if llm_provider == "None" else llm_provider
                final_llm_model = None if not llm_model or llm_provider == "None" else llm_model
                final_llm_api_key = None if not llm_api_key or llm_provider == "None" else llm_api_key
                final_llm_base_url = None if not llm_base_url or llm_provider == "None" else llm_base_url
                
                # Markdown conversation tracking
                agent_chat = []
                all_images = []
                
                # Preprocessing - giving execution feedback
                agent_chat.append("## 🚀 AI PhotoCrew started\n\n")
                agent_chat.append("### 📋 Photoshoot Details\n\n")
                agent_chat.append(f"- **Theme**: {topic}\n")
                agent_chat.append(f"- **Model**: {model}\n")
                agent_chat.append(f"- **Number of images**: {num_images}\n")
                agent_chat.append(f"- **Number of steps**: {steps}\n")
                agent_chat.append(f"- **LLM Provider**: {final_llm_provider if final_llm_provider else 'None'}\n")
                agent_chat.append(f"- **LLM Model**: {final_llm_model if final_llm_model else 'None'}\n\n")
                agent_chat.append("---\n\n")
                agent_chat.append("### 🏃‍♂️ PhotoCrew at work...\n\n")
                agent_chat.append("This is a fully automated process that goes through multiple iterations:\n\n")
                agent_chat.append("1. **Concept Phase**: Planning of the location and style\n")
                agent_chat.append("2. **First Generation**: Initial photos and analysis\n")
                agent_chat.append("3. **Second Generation**: Improved photos based on feedback\n")
                agent_chat.append("4. **Final Generation**: Final photos with all improvements\n\n")
                agent_chat.append("Each phase contains multiple steps and takes a few minutes. The entire process can take 10-15 minutes depending on the number of images.\n\n")
                
                # Yield initial state
                yield "Starting the PhotoCrew process...", all_images, "".join(agent_chat)
                
                try:
                    # Try to run the full crew first
                    print(f"Starting full PhotoCrew for: {topic}")
                    
                    # Launch the crew process
                    photo_crew = create_photo_crew(
                        topic, num_images, model, steps, lora_files, lora_scales,
                        final_llm_provider, final_llm_model, final_llm_api_key, final_llm_base_url
                    )
                    
                    # Run the crew with kickoff_with_updates to get streaming updates
                    result = ""
                    
                    # Instead of waiting for the entire process to complete
                    # Process the crew result line by line as it's running
                    for result_line in photo_crew.kickoff(yield_updates=True):
                        result += result_line + "\n"
                        
                        # Parse the result line
                        current_agent = None
                        current_generation = None
                        
                        # Check if this is the start of an agent's response
                        if result_line.startswith('# Agent:'):
                            current_agent = result_line.replace('# Agent:', '').strip()
                            agent_chat.append(f"\n### 🤖 {current_agent}\n")
                            
                            # Detect phase transitions based on agent
                            if current_agent == "Location Scout":
                                agent_chat.append(f"\n\n## 🏞️ Concept Phase Started\n\n")
                            elif current_agent == "Photographer" and "FIRST" in result_line:
                                agent_chat.append(f"\n\n## 📸 First Generation Phase Started\n\n")
                            elif current_agent == "Photographer" and "SECOND" in result_line:
                                agent_chat.append(f"\n\n## 📸 Second Generation Phase Started\n\n")
                            elif current_agent == "Photographer" and "FINAL" in result_line:
                                agent_chat.append(f"\n\n## 🏆 Final Generation Phase Started\n\n")
                        
                        # Check if this is an image path
                        elif result_line.startswith('/') and os.path.exists(result_line) and (result_line.endswith('.png') or result_line.endswith('.jpg')):
                            all_images.append(result_line)
                            
                            # Mark the image with the generation if we can determine it
                            if "FIRST VERSION" in result:
                                current_generation = "FIRST VERSION"
                            elif "SECOND VERSION" in result:
                                current_generation = "SECOND VERSION"
                            elif "FINAL VERSION" in result:
                                current_generation = "FINAL VERSION"
                            
                            if current_generation:
                                agent_chat.append(f"\n\n### 🖼️ {current_generation} of image generated\n\n")
                            else:
                                agent_chat.append(f"\n\n### 🖼️ Image generated\n\n")
                            
                            # Process images for the gallery
                            labeled_images = process_images_for_gallery(result, all_images, "".join(agent_chat))
                            
                            # Yield an update with the new image
                            yield result, labeled_images, "".join(agent_chat)
                        
                        # Regular content lines - add to the current agent's section
                        elif result_line and not result_line.startswith('#'):
                            agent_chat.append(result_line + "\n")
                            
                        # Yield regular updates to show progress
                        if len(result_line) > 0:
                            yield result, all_images, "".join(agent_chat)
                    
                    # At the end, wrap up
                    agent_chat.append("\n\n## ✅ Photoshoot Process Completed!\n\n")
                    agent_chat.append("The AI PhotoCrew has completed the creative process. ")
                    agent_chat.append("You can see the generated images in the gallery. ")
                    
                    yield result, all_images, "".join(agent_chat)
                except Exception as e:
                    error_msg = f"Error in running PhotoCrew: {str(e)}"
                    import traceback
                    tb = traceback.format_exc()
                    print(error_msg)
                    print(tb)
                    
                    # Try simplified crew as fallback in case of error
                    try:
                        agent_chat.append("\n\n### ⚠️ Error in full crew process\n\n")
                        agent_chat.append(f"There was a problem: {error_msg}\n\n")
                        agent_chat.append("We fall back to a simplified process without iterations...\n\n")
                        
                        # Yield an update about the fallback
                        yield f"Full crew failed, trying simplified crew: {error_msg}", all_images, "".join(agent_chat)
                        
                        # Run simplified crew
                        result = run_simplified_crew(
                            topic, num_images, model, steps, lora_files, lora_scales,
                            final_llm_provider, final_llm_model, final_llm_api_key, final_llm_base_url
                        )
                        
                        # Process results from simplified crew
                        if isinstance(result, str):
                            # Handle string results (backwards compatibility)
                            for line in result.split('\n'):
                                # Find image paths
                                if line.startswith('/') and os.path.exists(line) and (line.endswith('.png') or line.endswith('.jpg')):
                                    all_images.append(line)
                                    agent_chat.append(f"\n\n### 🖼️ Image generated\n\n")
                                    # Yield an update with the new image
                                    yield result, all_images, "".join(agent_chat)
                        else:
                            # Handle generator results (new streaming method)
                            for line in result:
                                # Add the line to the chat
                                agent_chat.append(line + "\n")
                                
                                # Check if this line contains an image path
                                if line.startswith('/') and os.path.exists(line) and (line.endswith('.png') or line.endswith('.jpg')):
                                    all_images.append(line)
                                    agent_chat.append(f"\n\n### 🖼️ Image generated\n\n")
                                    # Yield an update with the new image
                                    yield result, all_images, "".join(agent_chat)
                        
                        agent_chat.append("\n\n## ✅ Simplified Process Completed!\n\n")
                        agent_chat.append("The simplified PhotoCrew has generated images. ")
                        agent_chat.append("This process uses no iterations with feedback but still delivers results.")
                        
                        yield f"Simplified process completed (after error in full process): {result}", all_images, "".join(agent_chat)
                    except Exception as fallback_error:
                        fallback_tb = traceback.format_exc()
                        fallback_error_msg = f"Error in simplified process: {str(fallback_error)}"
                        print(fallback_error_msg)
                        print(fallback_tb)
                        
                        # If even the simplified crew fails, use the direct generator as last resort
                        try:
                            agent_chat.append("\n\n### ⚠️ Even the simplified process failed\n\n")
                            agent_chat.append(f"There was still a problem: {fallback_error_msg}\n\n")
                            agent_chat.append("We fall back to direct generation without CrewAI...\n\n")
                            
                            # Yield an update about the emergency fallback
                            yield f"Simplified crew failed, trying direct generation: {fallback_error_msg}", all_images, "".join(agent_chat)
                            
                            # Use direct generator as last option
                            result = generate_direct_image(
                                topic, num_images, model, steps, lora_files, lora_scales
                            )
                            
                            # Process results from direct generator
                            emergency_images = []
                            for line in result.split('\n'):
                                # Find image paths
                                if line.startswith('/') and os.path.exists(line) and (line.endswith('.png') or line.endswith('.jpg')):
                                    emergency_images.append(line)
                                    all_images.append(line)
                                    agent_chat.append(f"\n\n### 🖼️ Emergency image generated\n\n")
                                    
                                    # Yield an update with the new emergency image
                                    yield result, all_images, "".join(agent_chat)
                            
                            agent_chat.append("\n\n## ✅ Direct Generation Completed\n\n")
                            agent_chat.append("The direct generator has created images without AI-agents. ")
                            agent_chat.append("This process is limited but has delivered images.")
                            
                            yield f"Direct generation completed (after error in both crew processes): {result}", emergency_images, "".join(agent_chat)
                        except Exception as emergency_error:
                            emergency_tb = traceback.format_exc()
                            emergency_error_msg = f"Even direct generation failed: {str(emergency_error)}"
                            print(emergency_error_msg)
                            print(emergency_tb)
                            yield f"{error_msg}\n\nAlso fallback failed: {fallback_error_msg}\n\nAnd direct generation: {emergency_error_msg}", [], f"### ❌ All generation attempts failed\n\n{error_msg}\n\n```\n{tb}\n```\n\n### ❌ Also fallback failed\n\n{fallback_error_msg}\n\n```\n{fallback_tb}\n```\n\n### ❌ Also direct generation failed\n\n{emergency_error_msg}\n\n```\n{emergency_tb}\n```"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                import traceback
                tb = traceback.format_exc()
                print(error_msg)
                print(tb)
                yield f"{error_msg}\n\n{tb}", [], f"### ❌ Error occurred\n\n{error_msg}\n\n```\n{tb}\n```"

        # Wrapper function to convert image paths to (path, label) pairs
        def process_images_for_gallery(result_text, image_paths, chat):
            """Convert image paths to labeled gallery format for display"""
            
            # Create a list of (image_path, label) pairs
            labeled_images = []
            
            # If there are no images, return an empty list
            if not image_paths or len(image_paths) == 0:
                return []
            
            # Group images by generation (looking for patterns in filenames)
            first_gen = []
            second_gen = []
            final_gen = []
            
            for img_path in image_paths:
                if os.path.exists(img_path):
                    img_basename = os.path.basename(img_path)
                    
                    # Sort by generation and create labeled images
                    if "first" in img_path.lower() or "interpretation1" in img_path.lower():
                        label = f"First generation - {img_basename}"
                        first_gen.append((img_path, label))
                    elif "second" in img_path.lower() or "interpretation2" in img_path.lower():
                        label = f"Second generation - {img_basename}"
                        second_gen.append((img_path, label))
                    else:
                        label = f"Final version - {img_basename}"
                        final_gen.append((img_path, label))
            
            # Combine all labeled images, with most recent (final) first
            labeled_images = final_gen + second_gen + first_gen
            
            # Return the labeled images
            return labeled_images
        
        # Function to clear all
        def clear_results():
            return "Start a new photoshoot to see results...", [], "Agent conversations will appear here..."
        
        # Function to call gallery wrapper
        def wrapper_run_crew(*args):
            try:
                # Since run_crew is now a generator, we need to consume all values
                # and return the final state
                generator = run_crew(*args)
                result = None
                for result in generator:
                    pass
                
                # If we didn't get any results, return a default
                if result is None:
                    return "No results were generated.", [], "No agent conversations were recorded."
                
                # Return the final result (should be a tuple of 3 items)
                return result
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_msg = f"Error in wrapper_run_crew: {str(e)}\n\n{tb}"
                print(error_msg)
                return f"Error: {str(e)}", [], f"An error occurred:\n\n```\n{tb}\n```"
        
        # Gather all inputs for the click function
        inputs = [
            topic_input, num_images_input, model_dropdown, steps_slider,
            lora_files, llm_provider, llm_model, llm_api_key, llm_base_url
        ]
        inputs.extend([lora_scale for lora_scale in lora_scales])
        
        # Connect all functions to the interface
        start_button.click(
            fn=wrapper_run_crew,
            inputs=inputs,
            outputs=[crew_output, gallery, agent_conversations],
            queue=True,
            api_name="photocrew_stream"
        )
        
        clear_button.click(
            fn=clear_results,
            inputs=[],
            outputs=[crew_output, gallery, agent_conversations]
        )
    
    return photo_crew_tab 