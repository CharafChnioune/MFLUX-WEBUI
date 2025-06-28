import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

class DynamicPromptsManager:
    """Manager for dynamic prompts functionality from MFLUX v0.9.0"""
    
    def __init__(self):
        self.prompts_dir = Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        self.config_file = self.prompts_dir / "dynamic_prompts_config.json"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load dynamic prompts configuration"""
        default_config = {
            "enabled": True,
            "random_selection": True,
            "max_variations": 10,
            "replacement_markers": {
                "start": "[",
                "end": "]",
                "separator": "|"
            },
            "prompt_files": [],
            "categories": {
                "styles": [],
                "subjects": [],
                "environments": [],
                "lighting": [],
                "colors": [],
                "moods": []
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    default_config.update(saved_config)
            except Exception as e:
                print(f"Error loading dynamic prompts config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save dynamic prompts configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving dynamic prompts config: {e}")
    
    def load_prompt_file(self, file_path: Union[str, Path]) -> List[str]:
        """Load prompts from a file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'prompts' in data:
                        return data['prompts']
                    else:
                        return []
                else:
                    # Text file, one prompt per line
                    return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Error loading prompt file {file_path}: {e}")
            return []
    
    def save_prompt_file(self, prompts: List[str], file_path: Union[str, Path], format: str = "txt"):
        """Save prompts to a file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == "json":
                    json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)
                else:
                    f.write('\n'.join(prompts))
        except Exception as e:
            print(f"Error saving prompt file {file_path}: {e}")
    
    def parse_dynamic_prompt(self, prompt: str) -> List[str]:
        """Parse a dynamic prompt and return all possible variations"""
        if not self.config["enabled"]:
            return [prompt]
        
        start_marker = self.config["replacement_markers"]["start"]
        end_marker = self.config["replacement_markers"]["end"]
        separator = self.config["replacement_markers"]["separator"]
        
        # Find all dynamic parts in the prompt
        pattern = re.escape(start_marker) + r'([^' + re.escape(end_marker) + r']+)' + re.escape(end_marker)
        matches = re.findall(pattern, prompt)
        
        if not matches:
            return [prompt]
        
        # Generate variations
        variations = [prompt]
        
        for match in matches:
            options = [opt.strip() for opt in match.split(separator)]
            new_variations = []
            
            for variation in variations:
                for option in options:
                    new_variation = variation.replace(f"{start_marker}{match}{end_marker}", option, 1)
                    new_variations.append(new_variation)
            
            variations = new_variations
            
            # Limit variations to prevent explosion
            if len(variations) > self.config["max_variations"]:
                variations = random.sample(variations, self.config["max_variations"])
        
        return variations
    
    def get_random_prompt_variation(self, prompt: str) -> str:
        """Get a single random variation of a dynamic prompt"""
        variations = self.parse_dynamic_prompt(prompt)
        
        if self.config["random_selection"]:
            return random.choice(variations)
        else:
            return variations[0]  # Return first variation if not random
    
    def load_prompt_from_file(self, file_path: Union[str, Path]) -> str:
        """Load a random prompt from a file"""
        prompts = self.load_prompt_file(file_path)
        if not prompts:
            return ""
        
        return random.choice(prompts) if self.config["random_selection"] else prompts[0]
    
    def get_category_prompt(self, category: str) -> str:
        """Get a random prompt from a specific category"""
        if category not in self.config["categories"]:
            return ""
        
        category_prompts = self.config["categories"][category]
        if not category_prompts:
            return ""
        
        return random.choice(category_prompts) if self.config["random_selection"] else category_prompts[0]
    
    def add_to_category(self, category: str, prompts: List[str]):
        """Add prompts to a category"""
        if category not in self.config["categories"]:
            self.config["categories"][category] = []
        
        for prompt in prompts:
            if prompt not in self.config["categories"][category]:
                self.config["categories"][category].append(prompt)
        
        self.save_config()
    
    def remove_from_category(self, category: str, prompts: List[str]):
        """Remove prompts from a category"""
        if category not in self.config["categories"]:
            return
        
        for prompt in prompts:
            if prompt in self.config["categories"][category]:
                self.config["categories"][category].remove(prompt)
        
        self.save_config()
    
    def create_prompt_template(self, name: str, template: str, description: str = "") -> Path:
        """Create a prompt template file"""
        template_data = {
            "name": name,
            "description": description,
            "template": template,
            "created_at": str(Path(__file__).stat().st_mtime)
        }
        
        template_file = self.prompts_dir / f"template_{name.lower().replace(' ', '_')}.json"
        
        try:
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving template {template_file}: {e}")
        
        return template_file
    
    def load_prompt_template(self, template_name: str) -> Dict[str, Any]:
        """Load a prompt template"""
        template_file = self.prompts_dir / f"template_{template_name.lower().replace(' ', '_')}.json"
        
        if not template_file.exists():
            return {}
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading template {template_file}: {e}")
            return {}
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available prompt templates"""
        templates = []
        
        for template_file in self.prompts_dir.glob("template_*.json"):
            try:
                template_data = self.load_prompt_template(template_file.stem.replace("template_", ""))
                templates.append(template_data)
            except Exception as e:
                print(f"Error reading template {template_file}: {e}")
        
        return templates
    
    def get_available_prompt_files(self) -> List[str]:
        """Get list of available prompt files"""
        files = []
        
        for file in self.prompts_dir.glob("*.txt"):
            files.append(str(file))
        
        for file in self.prompts_dir.glob("*.json"):
            if not file.name.startswith("template_") and file.name != "dynamic_prompts_config.json":
                files.append(str(file))
        
        return files
    
    def enhance_prompt_with_categories(self, base_prompt: str, categories: List[str] = None) -> str:
        """Enhance a base prompt with random elements from categories"""
        if categories is None:
            categories = ["styles", "lighting", "colors"]
        
        enhanced_parts = [base_prompt]
        
        for category in categories:
            category_prompt = self.get_category_prompt(category)
            if category_prompt:
                enhanced_parts.append(category_prompt)
        
        return ", ".join(enhanced_parts)
    
    def create_wildcard_prompt(self, base_prompt: str, wildcards: Dict[str, List[str]]) -> str:
        """Create a dynamic prompt with wildcards"""
        dynamic_prompt = base_prompt
        
        for wildcard, options in wildcards.items():
            placeholder = f"[{wildcard}]"
            if placeholder in dynamic_prompt:
                options_str = "|".join(options)
                dynamic_prompt = dynamic_prompt.replace(placeholder, f"[{options_str}]")
        
        return self.get_random_prompt_variation(dynamic_prompt)
    
    def export_prompt_examples(self) -> str:
        """Export example prompts and templates"""
        examples = {
            "dynamic_prompts": [
                "A [beautiful|stunning|gorgeous] [cat|dog|bird] in a [garden|park|forest]",
                "[Realistic|Artistic|Stylized] portrait of a [young|old] [man|woman] with [blue|green|brown] eyes",
                "[Sunny|Cloudy|Rainy] day in [Tokyo|Paris|New York], [morning|afternoon|evening] light"
            ],
            "wildcard_examples": {
                "style": ["realistic", "artistic", "anime", "photographic", "painting"],
                "lighting": ["soft lighting", "dramatic lighting", "golden hour", "blue hour"],
                "mood": ["peaceful", "energetic", "mysterious", "romantic", "melancholic"]
            },
            "category_examples": {
                "styles": ["hyperrealistic", "oil painting", "watercolor", "digital art"],
                "subjects": ["portrait", "landscape", "still life", "abstract"],
                "environments": ["studio", "outdoor", "indoor", "natural"],
                "lighting": ["soft", "hard", "dramatic", "ambient"],
                "colors": ["vibrant", "muted", "monochrome", "pastel"],
                "moods": ["calm", "energetic", "mysterious", "joyful"]
            }
        }
        
        return json.dumps(examples, indent=2, ensure_ascii=False)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        self.save_config()

# Global instance
dynamic_prompts_manager = DynamicPromptsManager()

def get_dynamic_prompts_manager():
    """Get the global dynamic prompts manager instance"""
    return dynamic_prompts_manager

def process_dynamic_prompt(prompt: str) -> str:
    """Process a dynamic prompt and return a variation"""
    return dynamic_prompts_manager.get_random_prompt_variation(prompt)

def load_prompt_from_file(file_path: str) -> str:
    """Load a random prompt from file"""
    return dynamic_prompts_manager.load_prompt_from_file(file_path)
