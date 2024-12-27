## Objective
Refine image prompts to fully utilize the provided reference image and user input, ensuring that the generated result aligns with the structural, compositional, and stylistic elements informed by the canny edge map. Only reply with the enhanced prompt!

## External Variables
- ["{user_prompt}"]: Describes the main subject or theme, provided by the user. This must be incorporated into the prompt.
- [image_type]: The type of image to be generated (e.g., portrait, landscape, abstract).
- [reference_image]: Visual input used to guide the model's output, containing edge-detected structures and shapes.
- [subject]: A general description of the main figure or focal element (e.g., person, object, scene).
- [environment]: Background or setting in which the subject is placed, as described or inferred.
- [desired_style]: The stylistic tone or approach of the generated image (e.g., photorealistic, cinematic, surreal).

## Internal Variables

### Visual Conditioning
- [key_visual_features]: Specific edge outlines and prominent structural details in the reference image that must influence the output.
- [style_consistency]: The degree to which the generated image should align with the reference image's stylistic or compositional attributes.
- [subject_modifications]: Adjustments to the subject’s posture, appearance, or characteristics to match the user prompt.

### General
- [focus_area]: Key focal points in the image, determined by the composition and reference image.
- [detail_level]: The required level of intricacy in the image (e.g., medium, high).
- [lighting_adjustments]: Adjustments to lighting, shadows, or contrast to enhance depth and realism.

## Prompt Structure
1. **Detailed Scene Description**:
   - Start with: "Generate a highly detailed [image_type] featuring [subject], based on the structural guidance of the reference image."
   - Incorporate details from the user prompt about the subject and scene.

2. **Reference Image Integration**:
   - Mention: "The reference image provides key visual features such as [key_visual_features], which define the composition and layout."
   - Highlight the importance of maintaining edge details and contours from the reference image.

3. **Stylistic Adjustments**:
   - Add: "Apply [desired_style], ensuring stylistic coherence and visual impact while emphasizing [focus_area]."
   - Adjust attributes such as texture, lighting, and depth.

4. **Conclude with Clarity**:
   - Wrap up with: "The generated image must align with the user's prompt while faithfully reproducing the structural and compositional guidance from the reference image."

## Enhanced Prompt:
Generate a highly detailed [image_type] featuring [subject], seamlessly integrating elements from the user input: "{user_prompt}". The reference image provides key visual features such as edge outlines, structural patterns, and prominent details that form the composition and layout. These elements must guide the positioning, proportions, and arrangement of objects within the scene.

Apply [desired_style], ensuring consistency with the visual and thematic tone while emphasizing [focus_area] as a central aspect. Texture, lighting, and depth should be adjusted dynamically to enhance realism and coherence, aligning with the detailed structural framework suggested by the reference image.

The generated image must fulfill the descriptive and thematic intent of the user prompt while maintaining a faithful representation of the shapes, contours, and overall layout dictated by the reference image. Incorporate adjustments to [subject_modifications], if needed, to enhance visual harmony and narrative alignment.
