import os
from PIL import Image
from crewai.tools import BaseTool

class ImageAnalyzerTool(BaseTool):
    name: str = "Image Analyzer"
    description: str = "Analyzes an image and provides insights on composition, lighting, and style"
    
    def _run(self, image_path: str) -> str:
        """
        Analyzes an image and provides insights.
        
        Args:
            image_path: Path to the image to analyze
            
        Returns:
            Analysis of the image
        """
        try:
            print(f"Attempting to analyze image at: {image_path}")
            
            # Controleer of het bestand bestaat
            if not os.path.exists(image_path):
                # Probeer relatieve paden als het absolute pad niet werkt
                base_dir = os.getcwd()
                relative_path = os.path.basename(image_path)
                alternative_paths = [
                    relative_path,
                    os.path.join(base_dir, relative_path),
                    os.path.join(base_dir, "outputs", relative_path),
                    os.path.join(base_dir, "generated", relative_path)
                ]
                
                for alt_path in alternative_paths:
                    print(f"Trying alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        print(f"Found image at alternative path: {alt_path}")
                        image_path = alt_path
                        break
                
                if not os.path.exists(image_path):
                    return f"Image not found at path: {image_path} or any alternative locations"
                
            # Open de afbeelding
            img = Image.open(image_path)
            width, height = img.size
            
            # Converteer naar RGB als het een ander formaat is (zoals RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Krijg de gemiddelde RGB-waarden
            r, g, b = 0, 0, 0
            pixels = img.getdata()
            count = len(pixels)
            
            for pixel in pixels:
                r += pixel[0]
                g += pixel[1]
                b += pixel[2]
                
            r_avg = r // count
            g_avg = g // count
            b_avg = b // count
            
            # Eenvoudige analyse op basis van gemiddelde waarden
            brightness = (r_avg + g_avg + b_avg) / 3 / 255
            color_dominance = ""
            
            if r_avg > g_avg and r_avg > b_avg:
                color_dominance = "red tones"
            elif g_avg > r_avg and g_avg > b_avg:
                color_dominance = "green tones"
            elif b_avg > r_avg and b_avg > g_avg:
                color_dominance = "blue tones"
            else:
                color_dominance = "balanced color distribution"
                
            # Bestandsgrootte
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # In MB
            
            # Aspect ratio
            aspect_ratio = width / height
            orientation = "portrait" if aspect_ratio < 1 else "landscape" if aspect_ratio > 1 else "square"
            
            analysis = f"""
Image Analysis for {os.path.basename(image_path)}:
- Dimensions: {width}x{height} ({orientation} orientation)
- File size: {file_size:.2f} MB
- Brightness: {'bright' if brightness > 0.5 else 'dark'} ({brightness:.2f}/1.0)
- Color profile: Dominant {color_dominance}

Visual Assessment:
- The image has a {'high' if brightness > 0.6 else 'moderate' if brightness > 0.4 else 'low'} brightness level.
- It features prominent {color_dominance}.
- The composition is in {orientation} format with {aspect_ratio:.2f} aspect ratio.

Recommendations:
- {'Consider brightening the image.' if brightness < 0.4 else 'Consider toning down brightness.' if brightness > 0.7 else 'Brightness level is good.'}
- {'Consider balancing colors more evenly.' if max(r_avg, g_avg, b_avg) - min(r_avg, g_avg, b_avg) > 50 else 'Color balance is good.'}
"""
            return analysis
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

def analyze_image_tool():
    """Returns an initialized image analyzer tool"""
    return ImageAnalyzerTool() 