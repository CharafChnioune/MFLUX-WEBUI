## Objective
Enhance depth-guided image generation prompts by creating detailed descriptions that utilize depth map information to achieve accurate spatial relationships, realistic layering, and compelling depth perception in the generated images. Only reply with the enhanced prompt.

## External Variables
- [user_prompt]: The base description of the desired image
- [depth_reference]: Reference image providing depth map guidance
- [depth_emphasis]: Areas of specific depth interest (foreground, midground, background)
- [spatial_priority]: Primary focus areas within the depth hierarchy
- [scene_type]: Type of scene (landscape, interior, portrait, architectural, etc.)

## Internal Variables

### Depth Analysis
- [foreground_elements]: Objects and details in the closest depth layer
- [midground_composition]: Elements in the middle distance providing context
- [background_structure]: Distant elements creating depth and atmosphere
- [depth_transitions]: How layers blend and transition between depth planes
- [focal_depth]: Primary depth plane requiring sharpest detail and attention

### Spatial Relationships
- [perspective_accuracy]: Correct size relationships based on distance
- [atmospheric_perspective]: Haze, color shift, contrast changes with distance
- [overlapping_hierarchy]: How objects occlude others to create depth
- [shadow_depth_casting]: Shadows that enhance three-dimensional perception
- [lighting_depth_variation]: How light intensity changes across depth planes

### Visual Depth Cues
- [linear_perspective]: Converging lines and vanishing points
- [texture_gradient]: Texture detail reduction with distance
- [color_depth_shift]: Cooler, less saturated colors in distant areas
- [contrast_depth_variation]: Higher contrast in foreground, lower in background
- [detail_hierarchy]: Sharp detail foreground, softer detail background

### Depth-Specific Enhancements
- [spatial_storytelling]: How depth arrangement tells the scene's story
- [depth_of_field_simulation]: Focus effects that enhance depth perception
- [environmental_layering]: Natural depth created by environmental elements
- [architectural_depth]: Building and structural depth relationships

### Prompt Structure
- [prompt_starter]: "Create a depth-rich image with compelling spatial layers featuring "
- [depth_guidance]: Specific depth map utilization instructions
- [prompt_end_depth]: " with accurate perspective, realistic atmospheric depth, clear foreground-to-background progression, and compelling three-dimensional spatial relationships"

## Enhancement Process

1. **Analyze Depth Reference**:
   - Identify [foreground_elements], [midground_composition], and [background_structure]
   - Determine [focal_depth] and [spatial_priority] areas
   - Assess [depth_transitions] and [overlapping_hierarchy]

2. **Scene Type Optimization**:
   - **Landscape**: Emphasize atmospheric perspective and environmental layering
   - **Interior**: Focus on architectural depth and lighting depth variation
   - **Portrait**: Highlight subject separation and background bokeh effects
   - **Architectural**: Prioritize linear perspective and structural depth relationships

3. **Construct Enhanced Prompt**:
   - Begin with [prompt_starter]
   - Integrate [user_prompt] with [spatial_storytelling] elements
   - Add [perspective_accuracy] and [atmospheric_perspective] guidelines
   - Include [linear_perspective] and [texture_gradient] specifications
   - Incorporate [color_depth_shift] and [contrast_depth_variation]
   - Apply [shadow_depth_casting] and [lighting_depth_variation]
   - Add [detail_hierarchy] and [depth_of_field_simulation]
   - End with [prompt_end_depth]

4. **Depth Enhancement Strategies**:
   - **Strong Foreground**: Detailed, high-contrast elements that draw attention
   - **Rich Midground**: Context-providing elements that bridge fore and background
   - **Atmospheric Background**: Distant elements with reduced detail and contrast
   - **Transition Smoothness**: Natural blending between depth layers

5. **Technical Depth Considerations**:
   - Ensure depth map guidance is fully utilized
   - Maintain consistent perspective throughout all depth planes
   - Create believable atmospheric effects
   - Optimize for compelling three-dimensional perception

6. **Response Format**:
   Provide the comprehensive, depth-aware generation prompt without additional commentary.
