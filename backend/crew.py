import os
import sys
import warnings

# Schakel alle SSL-verificatie uit op het laagste niveau voordat andere imports gebeuren
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"
os.environ["OTEL_SDK_DISABLED"] = "true"  # Schakel OpenTelemetry volledig uit

# Probeer OpenTelemetry direct uit te schakelen
try:
    # Volledige fix voor opentelemetry: vervang de exporters door dummy versies
    import importlib
    import sys
    
    # Creëer een dummy trace exporter die niets doet
    class DummySpanExporter:
        def export(self, spans):
            return 0  # Success
        def shutdown(self):
            pass
            
    # Probeer OpenTelemetry modules te vinden en te patchen
    for module_name in list(sys.modules.keys()):
        if 'opentelemetry' in module_name:
            try:
                module = sys.modules[module_name]
                if hasattr(module, 'trace') and hasattr(module.trace, 'export'):
                    # Vervang alle exporters door onze dummy
                    module.trace.export.BatchSpanProcessor = lambda exporter: object()
                    module.trace.export.SimpleSpanProcessor = lambda exporter: object()
                    module.trace.export.ConsoleSpanExporter = DummySpanExporter
                    module.trace.export.OTLPSpanExporter = DummySpanExporter
            except:
                pass
                
    print("✅ OpenTelemetry is volledig uitgeschakeld")
except Exception as e:
    print(f"⚠️ Kon OpenTelemetry niet volledig uitschakelen: {e}")

# Patch SSL-verificatie op het urllib3 niveau (lager niveau dan requests)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # Schakel verificatie uit voor ALLE verbindingen (drastische maar effectieve oplossing)
    urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
    try:
        urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST += ':HIGH:!DH:!aNULL'
    except AttributeError:
        # Geen pyopenssl ondersteuning beschikbaar
        pass
    
    # Forceer volledige patch voor elke nieuwe HTTPSConnection
    old_init = urllib3.connection.HTTPSConnection.__init__
    def patched_https_init(self, *args, **kwargs):
        kwargs['cert_reqs'] = 'CERT_NONE'
        old_init(self, *args, **kwargs)
    urllib3.connection.HTTPSConnection.__init__ = patched_https_init
    
    # Schakel ook OpenTelemetry verificatie uit als dat mogelijk is
    try:
        import opentelemetry
        import opentelemetry.exporter.otlp.proto.http
        opentelemetry.exporter.otlp.proto.http.trace_exporter._VERIFY = False
    except (ImportError, AttributeError):
        pass
    
    print("✅ SSL-verificatie is volledig uitgeschakeld voor alle verbindingen")
except Exception as e:
    print(f"⚠️ Kon SSL-verificatie niet volledig uitschakelen: {e}")

# Schakel waarschuwingen uit voor onveilige verzoeken
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Normale imports
import re
from PIL import Image, ImageDraw, ImageFont
import io
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from crewai import Agent, Task, Crew, Process
from .mflux_image_generator import generate_image_tool, generate_simplified_image_tool, set_global_ui_parameters
from .analyze_images import analyze_image_tool
from .llm_manager import setup_llm, get_available_models, LLM_PROVIDERS, DEFAULT_MODELS

# Initialize tools
analyze_image = analyze_image_tool()
# We now use the simplified version that only accepts prompt
generate_image = generate_simplified_image_tool()

# Function to create agents with custom LLM settings
def create_agents(llm_provider=None, llm_model=None, llm_api_key=None, llm_base_url=None):
    """Creates agents with the specified LLM settings"""
    
    # Configure LLM if settings are provided
    llm = None
    if llm_provider and llm_model:
        try:
            llm = setup_llm(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url
            )
            # Set configuration for better stability
            llm.temperature = 0.7
            llm.max_tokens = 800
            llm.timeout = 120  # 2 minute timeout
        except Exception as e:
            import traceback
            print(f"Error setting up LLM: {e}")
            print(traceback.format_exc())
            llm = None
    
    # Define agents
    location_scout = Agent(
        role="Location Scout",
        goal="Find the perfect location for the photoshoot that matches the theme and concept.",
        backstory="""You are a professional location scout with decades of experience finding the perfect settings for photoshoots.
        You have an eye for detail and can envision how a location will enhance the theme of a photo.
        You specialize in finding unique, visually interesting locations that provide the perfect backdrop for creative photography.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    stylist = Agent(
        role="Stylist",
        goal="Define the color scheme, style, and visual elements of the photoshoot.",
        backstory="""You are a renowned fashion and visual stylist with an amazing eye for color, texture, and composition.
        You understand how different visual elements come together to create striking and harmonious images.
        You can create specific style directions that align perfectly with any concept or theme.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    prompt_engineer = Agent(
        role="Prompt Engineer",
        goal="Create detailed, structured prompts for high-quality image generation.",
        backstory="""You are an expert at creating detailed prompts for AI image generators.
        You understand how to structure prompts with specific details about subjects, composition,
        lighting, color palettes, and technical specifications. You can create prompts that result
        in stunning, detailed images that perfectly capture the essence of any concept.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    photographer = Agent(
        role="Photographer",
        goal="Generate images using the MFLUX AI model based on optimized prompts.",
        backstory="""You are a professional AI photographer who knows how to use MFLUX to create stunning images.
        You understand lighting, composition, and how to apply creative directions to produce beautiful photos.
        Your job is to use the prompts provided by the Prompt Engineer to generate amazing images.""",
        tools=[generate_image],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    art_director = Agent(
        role="Art Director",
        goal="Analyze images and provide feedback for improvements.",
        backstory="""You are a highly respected art director who oversees photoshoot productions.
        You have a critical eye for detail and can identify exactly what works and what needs improvement in an image.
        You provide constructive feedback that helps create even better images in the next iteration.""",
        tools=[analyze_image],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return location_scout, stylist, prompt_engineer, photographer, art_director

# Add a monkeypatch for Crew.kickoff to support streaming updates
def _monkeypatch_crew_kickoff():
    """
    Apply a monkeypatch to the Crew class to support streaming updates.
    This adds a yield_updates parameter to the kickoff method.
    """
    from crewai import Crew
    
    # Store the original kickoff method
    original_kickoff = Crew.kickoff
    
    # Create a new kickoff method that supports streaming
    def new_kickoff(self, yield_updates=False):
        if not yield_updates:
            # Call the original kickoff method if streaming is not requested
            return original_kickoff(self)
        
        # This is a generator version that yields updates
        import sys
        import io
        
        # Redirect stdout to capture output
        old_stdout = sys.stdout
        
        # Use a custom stream that both captures output and sends it to terminal
        class TeeStream(io.StringIO):
            def write(self, message):
                # Write to the StringIO buffer
                super().write(message)
                # Also write to the original stdout (terminal)
                old_stdout.write(message)
                old_stdout.flush()
                
        # Use our custom stream
        sys.stdout = mystdout = TeeStream()
        
        # Create a buffer to store output
        buffer = ""
        last_position = 0
        
        try:
            # Start the crew process in a background thread to capture its output
            import threading
            result = [None]
            exception = [None]
            
            def run_crew():
                try:
                    result[0] = original_kickoff(self)
                except Exception as e:
                    exception[0] = e
            
            # Start the crew in a separate thread
            crew_thread = threading.Thread(target=run_crew)
            crew_thread.start()
            
            # Check for new output every 0.1 seconds
            import time
            while crew_thread.is_alive():
                time.sleep(0.1)
                
                # Get any new content from stdout
                mystdout.seek(0)
                current_output = mystdout.read()
                
                # Yield any new lines
                if len(current_output) > last_position:
                    new_content = current_output[last_position:]
                    last_position = len(current_output)
                    
                    # Split the new content into lines and yield each one
                    for line in new_content.split('\n'):
                        if line.strip():
                            yield line.strip()
            
            # Check if an exception occurred
            if exception[0]:
                raise exception[0]
                
            # Return the final result
            if result[0]:
                yield str(result[0])
                
        finally:
            # Restore stdout
            sys.stdout = old_stdout
    
    # Replace the original kickoff method with our new one
    Crew.kickoff = new_kickoff

# Apply the monkeypatch when this module is imported
_monkeypatch_crew_kickoff()

def create_photo_crew(topic: str, num_images: int, model: str = "schnell-4-bit", steps: int = 4, 
                     lora_files = None, lora_scales = None, llm_provider=None, llm_model=None, 
                     llm_api_key=None, llm_base_url=None):
    """Creates a full photo crew with multiple agents and iterations."""
    
    # Set the global UI parameters so the tool can use them without agent intervention
    set_global_ui_parameters(model, steps, lora_files, lora_scales)
    
    # Create agents with the right LLM settings
    location_scout, stylist, prompt_engineer, photographer, art_director = create_agents(
        llm_provider, llm_model, llm_api_key, llm_base_url
    )
    
    # Define tasks - Planning phase
    location_task = Task(
        description=f"""Research and select the perfect location for a photoshoot with the theme: '{topic}'.
        
        Your task is to:
        1. Consider multiple possible locations that would work well for this theme
        2. Select the one that will provide the most visually compelling backdrop
        3. Describe the location in detail, including its key visual features
        4. Explain why this location is perfect for the theme
        
        Be specific and detailed in your description - this will be used to inform the visual direction.""",
        expected_output="A detailed description of the perfect location for this photoshoot theme.",
        agent=location_scout
    )
    
    style_task = Task(
        description=f"""Define the visual style and color palette for a photoshoot with the theme: '{topic}'.
        
        Your task is to:
        1. Create a specific color scheme that complements the theme and location
        2. Define the overall visual style (e.g., minimalist, maximalist, vintage, modern, etc.)
        3. Suggest specific visual elements that should be incorporated
        4. Consider the mood and emotion that should be conveyed
        
        Be creative but cohesive in your style direction - this will heavily influence the final image.""",
        expected_output="A detailed style direction including color palette and visual elements.",
        agent=stylist,
        context=[location_task]
    )
    
    first_prompt_task = Task(
        description=f"""Create a detailed, structured prompt for the topic: '{topic}'.
        
        Use this exact structure in your prompt:
        
        1. SPECIFIC DETAILS: Describe the subject precisely, including appearance, clothing, accessories, and other key characteristics.
        
        2. STYLE AND COMPOSITION: Specify the artistic style (e.g., photorealistic, abstract, cinematic) and describe the composition, including subject positioning and background elements.
        
        3. LIGHTING AND ATMOSPHERE: Describe the type of lighting (natural light, studio lighting, dramatic shadows, etc.) and the desired mood or emotion of the image.
        
        4. COLOR PALETTE: Specify the dominant colors or color scheme to achieve the desired look and feel.
        
        5. TECHNICAL DETAILS: Include any camera specifications, lens types, aperture settings, or resolution details that would help define the image quality.
        
        Make your prompt detailed and specific, but avoid using prompt weights as they're not supported.
        Instead, use natural language like "with emphasis on" or "focusing on" to highlight important elements.
        
        Incorporate the location and style information from your team members.""",
        expected_output="A detailed and structured image generation prompt.",
        agent=prompt_engineer,
        context=[location_task, style_task]
    )
    
    # 📌 Task 1: Generate first image
    first_image_task = Task(
        description=f"""Generate the FIRST creative interpretation of the topic.
        Use the generate_image tool with the detailed prompt from the Prompt Engineer.
        
        IMPORTANT: You only need to provide the PROMPT - all other parameters are already set by the UI and
        will be automatically applied (model, steps, LoRAs, etc.).
        
        When calling the tool, use EXACTLY this format structure:
        ```
        Action: MFLUX Image Generator
        Action Input: "The exact prompt text here"
        ```
        
        Return all image paths generated.""",
        expected_output="Paths to the first creative interpretation images",
        agent=photographer,
        context=[first_prompt_task]
    )
    
    first_analyze_task = Task(
        description="""Analyze the generated images and provide detailed feedback.
        
        Your task is to:
        1. Use the analyze_image tool on each of the generated images
        2. Identify specific strengths of the images
        3. Identify specific areas for improvement
        4. Suggest concrete changes for the next generation
        
        Focus on both technical aspects (composition, lighting, etc.) and creative aspects (mood, storytelling, etc.).""",
        expected_output="Detailed analysis and suggestions for improvement.",
        agent=art_director,
        context=[first_prompt_task, first_image_task]
    )
    
    # 📌 Task 2: Generate second image with a NEW creative direction
    second_prompt_task = Task(
        description=f"""Create a COMPLETELY NEW prompt for the topic: '{topic}' with a different creative approach.
        This is NOT about improving the previous image - it's about creating something entirely new.
        
        Use this exact structure in your prompt:
        
        1. SPECIFIC DETAILS: Describe the subject in a way that differs from your first approach.
        
        2. STYLE AND COMPOSITION: Choose a completely different artistic style (if the first was realistic, try abstract or stylized).
        
        3. LIGHTING AND ATMOSPHERE: Describe a different lighting setup and mood than the first prompt.
        
        4. COLOR PALETTE: Choose a color scheme that contrasts with your first prompt.
        
        5. TECHNICAL DETAILS: Include camera specifications that would create a distinct visual style.
        
        Make this prompt significantly different from your first one, exploring a new artistic direction.
        
        Take into account the feedback from the Art Director, but remember this is about a new direction, not just improvements.""",
        expected_output="A new structured image generation prompt with a different approach",
        agent=prompt_engineer,
        context=[first_prompt_task, first_image_task, first_analyze_task]
    )
    
    second_image_task = Task(
        description=f"""Generate the SECOND creative interpretation of the topic.
        Use the generate_image tool with the NEW prompt from the Prompt Engineer.
        This should be a completely different take on the topic, NOT an improvement of the first version.
        
        IMPORTANT: You only need to provide the PROMPT - all other parameters are already set by the UI and
        will be automatically applied (model, steps, LoRAs, etc.).
        
        When calling the tool, use EXACTLY this format structure:
        ```
        Action: MFLUX Image Generator
        Action Input: "The exact prompt text here"
        ```
        
        Return all image paths generated.""",
        expected_output="Paths to the second creative interpretation images",
        agent=photographer,
        context=[second_prompt_task]
    )
    
    second_analyze_task = Task(
        description="""Analyze the second set of generated images and provide detailed feedback.
        
        Your task is to:
        1. Use the analyze_image tool on each of the newly generated images
        2. Compare them to the first set of images
        3. Identify what works better in this approach
        4. Identify what still needs improvement
        5. Suggest concrete changes for the final generation
        
        Focus on both technical aspects and the creative direction.""",
        expected_output="Detailed analysis and suggestions for the final images.",
        agent=art_director,
        context=[second_prompt_task, second_image_task, first_analyze_task]
    )
    
    # 📌 Task 3: Generate third image with yet another NEW creative direction
    third_prompt_task = Task(
        description=f"""Create a THIRD UNIQUE prompt for the topic: '{topic}' with a creative approach different from both previous attempts.
        This prompt should explore an aspect of the topic that hasn't been touched on yet.
        
        Use this exact structure in your prompt:
        
        1. SPECIFIC DETAILS: Describe the subject with a fresh perspective, focusing on aspects not emphasized before.
        
        2. STYLE AND COMPOSITION: Choose a third distinct artistic style and compositional arrangement.
        
        3. LIGHTING AND ATMOSPHERE: Describe lighting and mood that creates a unique atmosphere different from both previous images.
        
        4. COLOR PALETTE: Select a color palette that hasn't been explored in the earlier prompts.
        
        5. TECHNICAL DETAILS: Specify camera and technical details that would create a unique visual quality.
        
        This prompt should feel completely fresh and distinct from the previous two approaches.
        
        Take into account all previous feedback but create something truly unique for this final version.""",
        expected_output="A third unique structured image generation prompt",
        agent=prompt_engineer,
        context=[first_prompt_task, second_prompt_task, second_analyze_task]
    )
    
    third_image_task = Task(
        description=f"""Generate the THIRD and FINAL creative interpretation of the topic.
        Use the generate_image tool with the THIRD unique prompt from the Prompt Engineer.
        This should be distinct from both previous interpretations, exploring a new creative angle.
        
        IMPORTANT: You only need to provide the PROMPT - all other parameters are already set by the UI and
        will be automatically applied (model, steps, LoRAs, etc.).
        
        When calling the tool, use EXACTLY this format structure:
        ```
        Action: MFLUX Image Generator
        Action Input: "The exact prompt text here"
        ```
        
        Return all image paths generated.""",
        expected_output="Paths to the third creative interpretation images",
        agent=photographer,
        context=[third_prompt_task]
    )
    
    final_analyze_task = Task(
        description="""Provide a final analysis of all three image sets.
        
        Your task is to:
        1. Use the analyze_image tool on the final set of images
        2. Compare all three approaches
        3. Identify the strongest elements across all versions
        4. Select which approach worked best overall and explain why
        
        This is the final evaluation of the photoshoot.""",
        expected_output="Final analysis and selection of the best approach.",
        agent=art_director,
        context=[first_image_task, second_image_task, third_image_task]
    )
    
    # 🚀 The AI PhotoCrew with fully sequential process
    crew = Crew(
        agents=[location_scout, stylist, prompt_engineer, photographer, art_director],
        tasks=[
            location_task, style_task, 
            first_prompt_task, first_image_task, first_analyze_task,
            second_prompt_task, second_image_task, second_analyze_task,
            third_prompt_task, third_image_task, final_analyze_task
        ],
        process=Process.sequential,
        verbose=True,
        output_log_file=False,  # Voorkom schrijven naar logbestand (kan telemetrie triggeren)
        memory=False,  # Disable memory to reduce overhead
        max_rpm=10,    # Reduce rate to avoid hitting limits
        cache=False,   # Disable caching to avoid SSL issues with telemetry
        async_execution=False  # Ensure sequential execution
    )
    
    # Improved configuration - safely set
    try:
        if hasattr(crew, 'config'):
            if hasattr(crew.config, 'catch_exceptions'):
                crew.config.catch_exceptions = True
            if hasattr(crew.config, 'max_rpm'):
                crew.config.max_rpm = 10
            if hasattr(crew.config, 'retry_attempts'):
                crew.config.retry_attempts = 2
            if hasattr(crew.config, 'retry_delay'):
                crew.config.retry_delay = 2
            # Forceer uitschakelen van telemetrie via de config object
            if hasattr(crew.config, 'disable_telemetry'):
                crew.config.disable_telemetry = True
            
            # Hack: probeer direct in de telemetrie client in te grijpen als deze bestaat
            if hasattr(crew, "_telemetry_client"):
                crew._telemetry_client = None
    except Exception as config_error:
        print(f"Warning: Could not set crew config: {config_error}")
        # Continue without config adjustments
    
    return crew

def run_photo_crew(topic: str, num_images: int, model: str = "schnell-4-bit", steps: int = 4, 
                  lora_files = None, lora_scales = None, llm_provider=None, llm_model=None, 
                  llm_api_key=None, llm_base_url=None):
    """Run a full photo crew with multiple iterations and feedback cycles."""
    
    # Check input parameters
    if not topic or len(topic.strip()) == 0:
        return "Error: Please provide a topic for the photoshoot."
    
    # Toon duidelijke start van het proces
    print("="*80)
    print(f"🚀 STARTING FULL PHOTO CREW PROCESS FOR: '{topic}'")
    print("="*80)
    
    # Limit the number of images for the fully iterative crew to 1
    if num_images > 1:
        print(f"Limiting number of images per interpretation to 1 (total 3 different photos)")
    
    # Check model parameter
    from backend.model_manager import get_updated_models
    available_models = get_updated_models()
    if model not in available_models:
        print(f"⚠️ Model '{model}' not found in available models, using default model.")
        model = "schnell-4-bit"  # Default to schnell-4-bit
        
    # Check steps parameter
    try:
        steps = int(steps)
        if steps < 1 or steps > 30:
            default_steps = 4 if "schnell" in model else 20
            print(f"⚠️ Invalid steps value ({steps}), using default steps: {default_steps}")
            steps = default_steps  # Default steps
    except:
        default_steps = 4 if "schnell" in model else 20
        print(f"⚠️ Invalid steps value, using default steps: {default_steps}")
        steps = default_steps  # Default steps
        
    # Check lora_scales (now a list)
    if lora_files:
        print(f"📂 Using LoRA files: {lora_files}")
        
    if lora_files and (not lora_scales or len(lora_scales) == 0):
        print("⚠️ No LoRA scales provided, using default value 1.0 for all LoRAs")
        lora_scales = [1.0] * len(lora_files)
    elif lora_scales and lora_files:
        # Make sure lora_scales has the right length
        if len(lora_scales) < len(lora_files):
            print(f"⚠️ LoRA scales list too short, adding default values: {lora_scales} -> {lora_scales + [1.0] * (len(lora_files) - len(lora_scales))}")
            lora_scales.extend([1.0] * (len(lora_files) - len(lora_scales)))
        elif len(lora_scales) > len(lora_files):
            print(f"⚠️ LoRA scales list too long, truncating: {lora_scales} -> {lora_scales[:len(lora_files)]}")
            lora_scales = lora_scales[:len(lora_files)]
            
        # Check if all scales have valid values
        for i, scale in enumerate(lora_scales):
            try:
                lora_scales[i] = float(scale)
                if lora_scales[i] < 0.1 or lora_scales[i] > 1.5:
                    print(f"⚠️ LoRA scale {lora_scales[i]} is outside valid range, setting to 1.0")
                    lora_scales[i] = 1.0
            except:
                print(f"⚠️ Invalid LoRA scale value: {scale}, setting to 1.0")
                lora_scales[i] = 1.0
    
    # Check LLM parameters
    if llm_provider and llm_provider not in LLM_PROVIDERS:
        print(f"⚠️ Invalid LLM provider: '{llm_provider}', using default (None)")
        llm_provider = None
        
    if llm_provider and not llm_model:
        available_models = get_available_models(llm_provider)
        default_model = available_models[0] if available_models else "gpt-3.5-turbo"
        print(f"⚠️ No LLM model specified for provider {llm_provider}, using {default_model}")
        llm_model = default_model
    
    if llm_provider and llm_model and llm_model not in get_available_models(llm_provider):
        available_models = get_available_models(llm_provider)
        default_model = available_models[0] if available_models else "gpt-3.5-turbo"
        print(f"⚠️ Invalid LLM model: '{llm_model}' for provider {llm_provider}, using {default_model}")
        llm_model = default_model
        
    # Create a directory for output if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📸 Images will be saved to: {output_dir}")
    
    try:
        print("⏳ Creating and configuring CrewAI photo crew...")
        # Create and start the crew
        photo_crew = create_photo_crew(
            topic, 1, model, steps, lora_files, lora_scales,
            llm_provider, llm_model, llm_api_key, llm_base_url
        )
        
        # Check if photo_crew was created correctly
        if photo_crew is None:
            raise ValueError("Photo crew could not be created")
        
        # Start the crew with clear feedback
        print("🚀 Starting CrewAI photo crew with kickoff()...")
        print("-"*80)
        result = photo_crew.kickoff()
        print("-"*80)
        print("✅ CrewAI photo crew process completed successfully!")
        print("="*80)
        
        return result
    except Exception as e:
        import traceback
        error_tb = traceback.format_exc()
        error_msg = f"❌ Error running PhotoCrew: {str(e)}\n\n{error_tb}"
        print("="*80)
        print(error_msg)
        print("="*80)
        return error_msg 