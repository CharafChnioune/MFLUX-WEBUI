import random
import json
import os
from typing import List, Dict, Optional

class AutoSeedsManager:
    """Manager for auto-seeds functionality from MFLUX v0.9.0"""
    
    def __init__(self):
        self.seeds_file = "configs/auto_seeds.json"
        self.auto_seeds = self.load_auto_seeds()
    
    def load_auto_seeds(self) -> Dict:
        """Load auto seeds configuration from file"""
        try:
            if os.path.exists(self.seeds_file):
                with open(self.seeds_file, 'r') as f:
                    return json.load(f)
            return {
                "enabled": False,
                "seed_pool": [],
                "current_index": 0,
                "shuffle": True,
                "exclude_seeds": []
            }
        except Exception as e:
            print(f"Error loading auto seeds: {e}")
            return {
                "enabled": False,
                "seed_pool": [],
                "current_index": 0,
                "shuffle": True,
                "exclude_seeds": []
            }
    
    def save_auto_seeds(self):
        """Save auto seeds configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.seeds_file), exist_ok=True)
            with open(self.seeds_file, 'w') as f:
                json.dump(self.auto_seeds, f, indent=2)
        except Exception as e:
            print(f"Error saving auto seeds: {e}")
    
    def generate_seed_pool(self, count: int = 100, min_seed: int = 1, max_seed: int = 2**32-1) -> List[int]:
        """Generate a pool of random seeds"""
        seed_pool = []
        for _ in range(count):
            while True:
                seed = random.randint(min_seed, max_seed)
                if seed not in self.auto_seeds["exclude_seeds"] and seed not in seed_pool:
                    seed_pool.append(seed)
                    break
        return seed_pool
    
    def add_seeds_to_pool(self, seeds: List[int]):
        """Add seeds to the auto seeds pool"""
        for seed in seeds:
            if seed not in self.auto_seeds["seed_pool"] and seed not in self.auto_seeds["exclude_seeds"]:
                self.auto_seeds["seed_pool"].append(seed)
        self.save_auto_seeds()
    
    def remove_seeds_from_pool(self, seeds: List[int]):
        """Remove seeds from the auto seeds pool"""
        for seed in seeds:
            if seed in self.auto_seeds["seed_pool"]:
                self.auto_seeds["seed_pool"].remove(seed)
        self.save_auto_seeds()
    
    def exclude_seeds(self, seeds: List[int]):
        """Add seeds to exclusion list"""
        for seed in seeds:
            if seed not in self.auto_seeds["exclude_seeds"]:
                self.auto_seeds["exclude_seeds"].append(seed)
                # Also remove from pool if present
                if seed in self.auto_seeds["seed_pool"]:
                    self.auto_seeds["seed_pool"].remove(seed)
        self.save_auto_seeds()
    
    def get_next_seed(self) -> Optional[int]:
        """Get the next seed from the auto seeds pool"""
        if not self.auto_seeds["enabled"] or not self.auto_seeds["seed_pool"]:
            return None
        
        if self.auto_seeds["shuffle"]:
            # Random selection from pool
            return random.choice(self.auto_seeds["seed_pool"])
        else:
            # Sequential selection
            current_index = self.auto_seeds["current_index"]
            if current_index >= len(self.auto_seeds["seed_pool"]):
                current_index = 0
            
            seed = self.auto_seeds["seed_pool"][current_index]
            self.auto_seeds["current_index"] = current_index + 1
            self.save_auto_seeds()
            return seed
    
    def enable_auto_seeds(self, enabled: bool):
        """Enable or disable auto seeds"""
        self.auto_seeds["enabled"] = enabled
        self.save_auto_seeds()
    
    def set_shuffle_mode(self, shuffle: bool):
        """Set shuffle mode for auto seeds"""
        self.auto_seeds["shuffle"] = shuffle
        self.save_auto_seeds()
    
    def get_pool_size(self) -> int:
        """Get current seed pool size"""
        return len(self.auto_seeds["seed_pool"])
    
    def clear_pool(self):
        """Clear the seed pool"""
        self.auto_seeds["seed_pool"] = []
        self.auto_seeds["current_index"] = 0
        self.save_auto_seeds()
    
    def reset_index(self):
        """Reset current index to 0"""
        self.auto_seeds["current_index"] = 0
        self.save_auto_seeds()
    
    def get_config(self) -> Dict:
        """Get current auto seeds configuration"""
        return self.auto_seeds.copy()
    
    def apply_auto_seed(self, current_seed: int) -> int:
        """Apply auto seed if enabled, otherwise return current seed"""
        if self.auto_seeds["enabled"]:
            auto_seed = self.get_next_seed()
            if auto_seed is not None:
                return auto_seed
        return current_seed

# Global instance
auto_seeds_manager = AutoSeedsManager()

def get_auto_seeds_manager():
    """Get the global auto seeds manager instance"""
    return auto_seeds_manager
