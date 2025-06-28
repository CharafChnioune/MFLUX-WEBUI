import psutil
import json
import os
from typing import Optional, Dict

class BatteryManager:
    """Manager for battery percentage stop limit functionality from MFLUX v0.9.0"""
    
    def __init__(self):
        self.config_file = "configs/battery_config.json"
        self.config = self.load_config()
        self.last_battery_check = 0
    
    def load_config(self) -> Dict:
        """Load battery configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return {
                "enabled": False,
                "stop_percentage": 20,
                "check_interval": 30,  # seconds
                "pause_on_low_battery": True,
                "resume_on_charge": True,
                "notification_enabled": True
            }
        except Exception as e:
            print(f"Error loading battery config: {e}")
            return {
                "enabled": False,
                "stop_percentage": 20,
                "check_interval": 30,
                "pause_on_low_battery": True,
                "resume_on_charge": True,
                "notification_enabled": True
            }
    
    def save_config(self):
        """Save battery configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving battery config: {e}")
    
    def get_battery_status(self) -> Optional[Dict]:
        """Get current battery status"""
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return None
            
            return {
                "percentage": battery.percent,
                "plugged": battery.power_plugged,
                "time_left": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
            }
        except Exception as e:
            print(f"Error getting battery status: {e}")
            return None
    
    def should_stop_generation(self) -> bool:
        """Check if generation should be stopped due to low battery"""
        if not self.config["enabled"]:
            return False
        
        battery_status = self.get_battery_status()
        if battery_status is None:
            return False  # Can't determine battery status, continue
        
        # Stop if battery is below threshold and not plugged in
        if (battery_status["percentage"] <= self.config["stop_percentage"] and 
            not battery_status["plugged"]):
            if self.config["notification_enabled"]:
                print(f"⚠️  Battery low ({battery_status['percentage']}%), stopping generation to preserve battery")
            return True
        
        return False
    
    def should_pause_generation(self) -> bool:
        """Check if generation should be paused due to battery settings"""
        if not self.config["enabled"] or not self.config["pause_on_low_battery"]:
            return False
        
        battery_status = self.get_battery_status()
        if battery_status is None:
            return False
        
        # Pause if battery is low but above stop threshold
        pause_threshold = self.config["stop_percentage"] + 10  # 10% buffer above stop
        if (battery_status["percentage"] <= pause_threshold and 
            not battery_status["plugged"]):
            return True
        
        return False
    
    def can_resume_generation(self) -> bool:
        """Check if generation can be resumed (e.g., when charging)"""
        if not self.config["enabled"] or not self.config["resume_on_charge"]:
            return True  # Always allow if not configured
        
        battery_status = self.get_battery_status()
        if battery_status is None:
            return True  # Can't determine status, allow
        
        # Resume if plugged in or battery is above safe threshold
        safe_threshold = self.config["stop_percentage"] + 15
        return (battery_status["plugged"] or 
                battery_status["percentage"] > safe_threshold)
    
    def update_config(self, **kwargs):
        """Update battery configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        self.save_config()
    
    def enable_battery_monitoring(self, enabled: bool):
        """Enable or disable battery monitoring"""
        self.config["enabled"] = enabled
        self.save_config()
    
    def set_stop_percentage(self, percentage: int):
        """Set battery percentage at which to stop generation"""
        if 5 <= percentage <= 50:  # Reasonable range
            self.config["stop_percentage"] = percentage
            self.save_config()
    
    def set_check_interval(self, interval: int):
        """Set battery check interval in seconds"""
        if 10 <= interval <= 300:  # 10 seconds to 5 minutes
            self.config["check_interval"] = interval
            self.save_config()
    
    def get_battery_info_string(self) -> str:
        """Get formatted battery information string"""
        battery_status = self.get_battery_status()
        if battery_status is None:
            return "Battery status unavailable"
        
        status_parts = [
            f"Battery: {battery_status['percentage']}%",
            "Plugged in" if battery_status["plugged"] else "On battery"
        ]
        
        if battery_status["time_left"]:
            hours = battery_status["time_left"] // 3600
            minutes = (battery_status["time_left"] % 3600) // 60
            status_parts.append(f"Time left: {hours}h {minutes}m")
        
        if self.config["enabled"]:
            status_parts.append(f"Stop at: {self.config['stop_percentage']}%")
        
        return " | ".join(status_parts)
    
    def get_config(self) -> Dict:
        """Get current battery configuration"""
        return self.config.copy()

# Global instance
battery_manager = BatteryManager()

def get_battery_manager():
    """Get the global battery manager instance"""
    return battery_manager

def check_battery_before_generation() -> bool:
    """Check if it's safe to start generation based on battery"""
    return not battery_manager.should_stop_generation()

def monitor_battery_during_generation() -> Dict[str, bool]:
    """Monitor battery during generation and return status"""
    return {
        "should_stop": battery_manager.should_stop_generation(),
        "should_pause": battery_manager.should_pause_generation(),
        "can_resume": battery_manager.can_resume_generation(),
        "battery_info": battery_manager.get_battery_info_string()
    }
