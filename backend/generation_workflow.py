"""
Generation workflow integration for MFLUX WebUI v0.15.4
Integrates shared features like auto-seeds, dynamic prompts, etc.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from backend.auto_seeds_manager import get_auto_seeds_manager
from backend.dynamic_prompts_manager import get_dynamic_prompts_manager
from backend.config_manager import get_config_manager


class GenerationWorkflow:
    """Manages the complete generation workflow with shared features"""
    
    def __init__(self):
        self.auto_seeds_manager = get_auto_seeds_manager()
        self.dynamic_prompts_manager = get_dynamic_prompts_manager()
        self.config_manager = get_config_manager()
        
        self.is_paused = False
        self.should_stop = False
        self.generation_stats = {
            "total_generated": 0,
            "failed_generations": 0,
        }
    
    def pre_generation_checks(self) -> Dict[str, Any]:
        """Run pre-generation checks and return status"""
        checks = {
            "can_proceed": True,
            "warnings": [],
            "errors": [],
            "config_valid": True
        }
        
        # Config validation only (battery logic removed)
        try:
            current_config = self.config_manager.get_current_config()
            validation_errors = self.config_manager.validate_config(current_config)
            if validation_errors:
                checks["config_valid"] = False
                checks["warnings"].extend(validation_errors)
        except Exception as e:
            checks["warnings"].append(f"Config validation error: {str(e)}")
        
        return checks
    
    def process_prompt(self, prompt: str, enable_dynamic: bool = True) -> str:
        """Process prompt with dynamic prompts if enabled"""
        if not enable_dynamic or not self.dynamic_prompts_manager.get_config()["enabled"]:
            return prompt
        
        try:
            # Check if prompt contains dynamic elements
            if '[' in prompt and '|' in prompt and ']' in prompt:
                processed_prompt = self.dynamic_prompts_manager.get_random_prompt_variation(prompt)
                return processed_prompt
            else:
                return prompt
        except Exception as e:
            print(f"Warning: Dynamic prompt processing failed: {e}")
            return prompt
    
    def get_seed_for_generation(self, provided_seed: Optional[int] = None) -> int:
        """Get seed for generation, considering auto-seeds"""
        auto_seeds_config = self.auto_seeds_manager.get_config()
        
        # If auto-seeds is enabled and no specific seed provided
        if auto_seeds_config["enabled"] and provided_seed is None:
            try:
                return self.auto_seeds_manager.get_next_seed()
            except Exception as e:
                print(f"Warning: Auto-seeds failed, using random seed: {e}")
                import random
                return random.randint(0, 2**32 - 1)
        
        # Use provided seed or generate random
        if provided_seed is not None:
            return provided_seed
        else:
            import random
            return random.randint(0, 2**32 - 1)
    
    def monitor_generation_progress(self, step: int, total_steps: int) -> Dict[str, Any]:
        """Monitor generation progress and check for interruptions"""
        return {
            "should_continue": True,
            "should_pause": False,
            "step": step,
            "total_steps": total_steps,
            "progress": step / total_steps if total_steps > 0 else 0
        }
    
    def handle_generation_pause(self, pause_duration: int = 30):
        """Handle generation pause (no-op without battery logic)"""
        if self.is_paused:
            time.sleep(pause_duration)
            self.is_paused = False
    
    def save_generation_metadata(self, image_path: Path, metadata: Dict[str, Any]):
        """Save enhanced metadata with v0.9.0 features"""
        if not self.config_manager.get_config_value("generation.save_metadata", True):
            return
        
        try:
            # Enhanced metadata
            enhanced_metadata = {
                **metadata,
                "mflux_version": "0.15.4",
                "generation_timestamp": time.time(),
                "generation_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "auto_seeds_enabled": self.auto_seeds_manager.get_config()["enabled"],
                "dynamic_prompts_enabled": self.dynamic_prompts_manager.get_config()["enabled"],
                "workflow_stats": self.generation_stats.copy()
            }
            
            # Save metadata file
            metadata_path = image_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
    
    def update_statistics(self, success: bool = True):
        """Update generation statistics"""
        if success:
            self.generation_stats["total_generated"] += 1
        else:
            self.generation_stats["failed_generations"] += 1
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "is_paused": self.is_paused,
            "should_stop": self.should_stop,
            "stats": self.generation_stats.copy(),
            "auto_seeds_enabled": self.auto_seeds_manager.get_config()["enabled"],
            "dynamic_prompts_enabled": self.dynamic_prompts_manager.get_config()["enabled"]
        }
    
    def reset_workflow_state(self):
        """Reset workflow state for new generation batch"""
        self.is_paused = False
        self.should_stop = False


# Global workflow instance
_workflow_instance = None

def get_generation_workflow() -> GenerationWorkflow:
    """Get the global generation workflow instance"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = GenerationWorkflow()
    return _workflow_instance


# Helper functions for easy integration

def check_pre_generation() -> Dict[str, Any]:
    """Quick pre-generation check"""
    workflow = get_generation_workflow()
    return workflow.pre_generation_checks()

def process_dynamic_prompt(prompt: str) -> str:
    """Process prompt with dynamic prompts"""
    workflow = get_generation_workflow()
    return workflow.process_prompt(prompt)

def get_next_seed(provided_seed: Optional[int] = None) -> int:
    """Get next seed considering auto-seeds"""
    workflow = get_generation_workflow()
    return workflow.get_seed_for_generation(provided_seed)

def monitor_step_progress(step: int, total_steps: int) -> Dict[str, Any]:
    """Monitor generation step progress"""
    workflow = get_generation_workflow()
    return workflow.monitor_generation_progress(step, total_steps)

def save_enhanced_metadata(image_path: Path, metadata: Dict[str, Any]):
    """Save enhanced metadata"""
    workflow = get_generation_workflow()
    workflow.save_generation_metadata(image_path, metadata)

def update_generation_stats(success: bool = True):
    """Update generation statistics"""
    workflow = get_generation_workflow()
    workflow.update_statistics(success)
