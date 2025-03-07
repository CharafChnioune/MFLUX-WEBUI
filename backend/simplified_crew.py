"""
Simplified version of the PhotoCrew with three different photo interpretations.
"""

import os
import sys
import warnings

# Schakel alle SSL-verificatie uit op het laagste niveau voordat andere imports gebeuren
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"

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
    
    print("✅ SSL-verificatie is volledig uitgeschakeld voor alle verbindingen in simplified_crew")
except Exception as e:
    print(f"⚠️ Kon SSL-verificatie niet volledig uitschakelen in simplified_crew: {e}")

# Schakel waarschuwingen uit voor onveilige verzoeken
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Normale imports
from typing import List, Optional, Union

from crewai import Agent, Task, Crew, Process
from .mflux_image_generator import generate_simplified_image_tool, set_global_ui_parameters
from .analyze_images import analyze_image_tool
from .llm_manager import setup_llm, get_available_models, LLM_PROVIDERS, DEFAULT_MODELS

# Initialize tools
generate_image = generate_simplified_image_tool()
analyze_image = analyze_image_tool()

def create_simplified_crew(topic: str, num_images: int, model: str = "schnell-4-bit", steps: int = 4, 
                     lora_files = None, lora_scales = None, llm_provider=None, llm_model=None, 
                     llm_api_key=None, llm_base_url=None):
    """Creates a simplified photo crew that generates three different photo interpretations."""
    
    # Set the global UI parameters so the tool can use them without agent intervention
    set_global_ui_parameters(model, steps, lora_files, lora_scales)
    
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
        goal="Generate diverse creative interpretations using the MFLUX AI model.",
        backstory="""You are a professional AI photographer who uses MFLUX to create stunning images.
        You understand composition, lighting, and how to apply creative prompts to produce
        varied and interesting visual interpretations of a concept.""",
        tools=[generate_image],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # First prompt task - structured according to guidelines
    first_prompt_task = Task(
        description=f"""Create a detailed, structured prompt for the topic: '{topic}'.
        
        Use this exact structure in your prompt:
        
        1. SPECIFIC DETAILS: Describe the subject precisely, including appearance, clothing, accessories, and other key characteristics.
        
        2. STYLE AND COMPOSITION: Specify the artistic style (e.g., photorealistic, abstract, cinematic) and describe the composition, including subject positioning and background elements.
        
        3. LIGHTING AND ATMOSPHERE: Describe the type of lighting (natural light, studio lighting, dramatic shadows, etc.) and the desired mood or emotion of the image.
        
        4. COLOR PALETTE: Specify the dominant colors or color scheme to achieve the desired look and feel.
        
        5. TECHNICAL DETAILS: Include any camera specifications, lens types, aperture settings, or resolution details that would help define the image quality.
        
        Make your prompt detailed and specific, but avoid using prompt weights as they're not supported.
        Instead, use natural language like "with emphasis on" or "focusing on" to highlight important elements.""",
        expected_output="A detailed and structured image generation prompt following the 5-part structure.",
        agent=prompt_engineer
    )
    
    # First image task
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
    
    # Second prompt task for a DIFFERENT interpretation
    second_prompt_task = Task(
        description=f"""Create a COMPLETELY NEW prompt for the topic: '{topic}' with a different creative approach.
        This is NOT about improving the previous image - it's about creating something entirely new.
        
        Use this exact structure in your prompt:
        
        1. SPECIFIC DETAILS: Describe the subject in a way that differs from your first approach.
        
        2. STYLE AND COMPOSITION: Choose a completely different artistic style (if the first was realistic, try abstract or stylized).
        
        3. LIGHTING AND ATMOSPHERE: Describe a different lighting setup and mood than the first prompt.
        
        4. COLOR PALETTE: Choose a color scheme that contrasts with your first prompt.
        
        5. TECHNICAL DETAILS: Include camera specifications that would create a distinct visual style.
        
        Make this prompt significantly different from your first one, exploring a new artistic direction.""",
        expected_output="A second structured image generation prompt with a different approach",
        agent=prompt_engineer,
        context=[first_prompt_task]
    )
    
    # Second image task
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
    
    # Third prompt task for yet another DIFFERENT interpretation
    third_prompt_task = Task(
        description=f"""Create a THIRD UNIQUE prompt for the topic: '{topic}' with a creative approach different from both previous attempts.
        This prompt should explore an aspect of the topic that hasn't been touched on yet.
        
        Use this exact structure in your prompt:
        
        1. SPECIFIC DETAILS: Describe the subject with a fresh perspective, focusing on aspects not emphasized before.
        
        2. STYLE AND COMPOSITION: Choose a third distinct artistic style and compositional arrangement.
        
        3. LIGHTING AND ATMOSPHERE: Describe lighting and mood that creates a unique atmosphere different from both previous images.
        
        4. COLOR PALETTE: Select a color palette that hasn't been explored in the earlier prompts.
        
        5. TECHNICAL DETAILS: Specify camera and technical details that would create a unique visual quality.
        
        This prompt should feel completely fresh and distinct from the previous two approaches.""",
        expected_output="A third unique structured image generation prompt",
        agent=prompt_engineer,
        context=[first_prompt_task, second_prompt_task]
    )
    
    # Third image task
    third_image_task = Task(
        description=f"""Generate the THIRD creative interpretation of the topic.
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
    
    # Create and configure the crew
    crew = Crew(
        agents=[prompt_engineer, photographer],
        tasks=[
            first_prompt_task, first_image_task,
            second_prompt_task, second_image_task,
            third_prompt_task, third_image_task
        ],
        process=Process.sequential,
        verbose=True,
        output_log_file=False,  # Voorkom schrijven naar logbestand (kan telemetrie triggeren)
        memory=False,  # Disable memory to reduce overhead
        max_rpm=10,    # Reduce rate to avoid hitting limits
        cache=False,   # Disable caching to avoid SSL issues with telemetry
        async_execution=False  # Ensure sequential execution
    )
    
    # Enhanced configuration - safely set
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
        print(f"Warning: Could not set simplified crew config: {config_error}")
        # Continue without config adjustments
    
    return crew

def run_simplified_crew(topic: str, num_images: int, model: str = "schnell-4-bit", steps: int = 4, 
                  lora_files = None, lora_scales = None, llm_provider=None, llm_model=None, 
                  llm_api_key=None, llm_base_url=None):
    """Run a simplified photo crew with multiple image generations."""
    
    # Validation
    if not topic or len(topic.strip()) == 0:
        return "Error: Please provide a topic for the photoshoot."
    
    # Toon duidelijke start van het proces
    print("="*80)
    print(f"🚀 STARTING SIMPLIFIED PHOTO CREW PROCESS FOR: '{topic}'")
    print("="*80)
    
    # Validate num_images
    try:
        num_images = int(num_images)
        if num_images < 1 or num_images > 10:
            print(f"⚠️ Invalid number of images ({num_images}), using default value: 3")
            num_images = 3  # Default value
    except:
        print("⚠️ Invalid number of images, using default value: 3")
        num_images = 3  # Default value
    
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
    
    # Check lora_files and lora_scales
    if lora_files:
        print(f"📂 Using LoRA files: {lora_files}")
        
    if lora_files and (not lora_scales or len(lora_scales) == 0):
        print("⚠️ No LoRA scales provided, using default value 1.0 for all LoRAs")
        lora_scales = [1.0] * len(lora_files)
    elif lora_scales and lora_files:
        # Make sure lora_scales has the right length
        if len(lora_scales) < len(lora_files):
            print(f"⚠️ LoRA scales list too short, adding default values")
            lora_scales.extend([1.0] * (len(lora_files) - len(lora_scales)))
        elif len(lora_scales) > len(lora_files):
            print(f"⚠️ LoRA scales list too long, truncating")
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
        print("⏳ Creating and configuring simplified CrewAI photo crew...")
        # Create and start the crew
        photo_crew = create_simplified_crew(
            topic, num_images, model, steps, lora_files, lora_scales,
            llm_provider, llm_model, llm_api_key, llm_base_url
        )
        
        # Check if photo_crew was created correctly
        if photo_crew is None:
            raise ValueError("Simplified photo crew could not be created")
        
        # Start the crew with clear feedback
        print("🚀 Starting simplified CrewAI photo crew with kickoff()...")
        print("-"*80)
        result = photo_crew.kickoff()
        print("-"*80)
        print("✅ Simplified CrewAI photo crew process completed successfully!")
        print("="*80)
        
        return result
    except Exception as e:
        import traceback
        error_tb = traceback.format_exc()
        error_msg = f"❌ Error running simplified PhotoCrew: {str(e)}\n\n{error_tb}"
        print("="*80)
        print(error_msg)
        print("="*80)
        return error_msg