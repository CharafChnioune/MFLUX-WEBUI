## Objective
Enhance the incoming user prompt "{user_prompt}" and use the image as a reference with the variables and the prompt structure.

## External Variables
- [image_type]
- [subject]
- [environment]
- [subject_details]
- [weather]
- [orientation]
- [artistic_influence]

## Internal Variables

### Photography-specific
- [camera]: If [image_type] is a photo, choose an appropriate camera model (e.g., Nikon D850)
- [camera_lens]: If [image_type] is a photo, select a suitable lens type (e.g., wide-angle lens)
- [camera_settings]: If [image_type] is a photo, choose optimal camera settings (ISO, shutter speed, depth of field)
- [photo_color_style]: If [image_type] is a photo, decide on a color style (e.g., natural, vibrant)
- [photographer]: If [image_type] is a photo, you may reference a famous photographer for style

### Art-specific
- [art_style]: If [image_type] is art, select an art style (e.g., impressionism, concept art)
- [paint_style]: If [image_type] is art, choose a paint style (e.g., oil painting with thick brush strokes)
- [artist]: If [image_type] is art, you may reference a famous artist for style

### General
- [mood]: Determine a dominant mood based on the [subject] and [environment]
- [model]: Build a detailed description of the [subject] using [subject_details]
- [shot_factors]: Based on the [environment], choose background focal points

### Prompt Structure
- [prompt_starter]: "Ultra High Resolution [image_type] of "
- [prompt_end_part1]: " award-winning, epic composition, ultra detailed."

## Additional Variables
- [subject_environment]: The environment best suited for the [subject]
- [subjects_detail_specific]: Specific details best suited for the [subject] (e.g., a 20-year-old female with blonde hair wearing a red dress)
- [subjects_weatherOrLights_Specific]: Weather or lighting that complements the [subject] and [environment]

## Enhancement Process

1. **Extract Details**: 
   Analyze the incoming prompt and extract relevant information to populate the external variables.

2. **Determine Internal Variables**: 
   Based on the external variables, assign appropriate values to the internal variables.

3. **Construct the Enhanced Prompt**:
   - Begin with [prompt_starter]
   - Incorporate [model], including [subjects_detail_specific]
   - Describe the [environment] in detail, incorporating [subject_environment] and [shot_factors]
   - Include details about [weather] or [subjects_weatherOrLights_Specific]
   - If applicable, mention the [camera], [camera_lens], and [camera_settings]
   - Reference the [artistic_influence], [photographer], or [artist] if provided
   - Convey the [mood] throughout the description
   - Use vivid language to describe textures, lighting, movements, reflections, and shadows
   - Insert [prompt_end_part1] just before the end
   - Do not end with a period

4. **Response Format**: 
   Provide the fully constructed, detailed prompt without any additional comments or preambles.

## Enhanced Prompt:
Ultra High Resolution [image_type] of [subject], a meticulously crafted depiction featuring [subjects_detail_specific], set in [environment] with [subject_environment] elements and [shot_factors]. Capturing the scene under [weather] conditions with [subjects_weatherOrLights_Specific] lighting that enhances the composition's mood. The style is informed by [artistic_influence] and employs [art_style] with [paint_style], or photography using [camera] with [camera_lens] and [camera_settings]. The result should evoke [mood], with rich textures, lifelike reflections, dynamic movements, and atmospheric shadows, resulting in an [photo_color_style] palette. award-winning, epic composition, ultra detailed
