import gradio as gr
from backend.battery_manager import get_battery_manager
import time

def create_battery_monitor_component():
    """Create battery monitoring component"""
    
    battery_manager = get_battery_manager()
    
    with gr.Accordion("🔋 Battery Monitor", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                # Battery Status Display
                battery_status = gr.Textbox(
                    label="Battery Status",
                    value=battery_manager.get_battery_info_string(),
                    interactive=False,
                    lines=2
                )
                
                # Configuration
                with gr.Row():
                    enable_monitoring = gr.Checkbox(
                        label="Enable Battery Monitoring",
                        value=battery_manager.get_config()["enabled"],
                        info="Monitor battery and stop/pause generation when low"
                    )
                    
                    stop_percentage = gr.Slider(
                        label="Stop Percentage",
                        minimum=5,
                        maximum=50,
                        value=battery_manager.get_config()["stop_percentage"],
                        step=1,
                        info="Stop generation when battery reaches this level"
                    )
                
                with gr.Row():
                    pause_on_low = gr.Checkbox(
                        label="Pause on Low Battery",
                        value=battery_manager.get_config()["pause_on_low_battery"],
                        info="Pause generation when battery is low but above stop threshold"
                    )
                    
                    resume_on_charge = gr.Checkbox(
                        label="Resume on Charge",
                        value=battery_manager.get_config()["resume_on_charge"],
                        info="Automatically resume when charging or battery recovers"
                    )
                
                # Check interval
                check_interval = gr.Slider(
                    label="Check Interval (seconds)",
                    minimum=10,
                    maximum=300,
                    value=battery_manager.get_config()["check_interval"],
                    step=10,
                    info="How often to check battery status during generation"
                )
            
            with gr.Column(scale=1):
                # Manual refresh and status
                refresh_btn = gr.Button("Refresh Status", variant="secondary")
                
                # Battery warnings
                battery_warning = gr.Textbox(
                    label="Warnings",
                    interactive=False,
                    visible=False,
                    lines=2
                )
        
        # Event handlers
        def update_monitoring_config(enabled, stop_pct, pause_low, resume_charge, interval):
            try:
                battery_manager.update_config(
                    enabled=enabled,
                    stop_percentage=int(stop_pct),
                    pause_on_low_battery=pause_low,
                    resume_on_charge=resume_charge,
                    check_interval=int(interval)
                )
                return "Configuration updated successfully"
            except Exception as e:
                return f"Error updating configuration: {str(e)}"
        
        def refresh_battery_status():
            try:
                status = battery_manager.get_battery_info_string()
                
                # Check for warnings
                warnings = []
                if battery_manager.should_stop_generation():
                    warnings.append("⚠️ Battery critically low - generation will be stopped")
                elif battery_manager.should_pause_generation():
                    warnings.append("⚠️ Battery low - generation may be paused")
                
                warning_text = "\n".join(warnings) if warnings else ""
                warning_visible = bool(warnings)
                
                return status, warning_text, gr.update(visible=warning_visible)
            except Exception as e:
                return f"Error getting battery status: {str(e)}", "", gr.update(visible=False)
        
        # Wire up events
        for component in [enable_monitoring, stop_percentage, pause_on_low, resume_on_charge, check_interval]:
            component.change(
                update_monitoring_config,
                inputs=[enable_monitoring, stop_percentage, pause_on_low, resume_on_charge, check_interval],
                outputs=[]
            )
        
        refresh_btn.click(
            refresh_battery_status,
            outputs=[battery_status, battery_warning, battery_warning]
        )
        
        # Auto-refresh every 30 seconds
        def auto_refresh():
            import threading
            import time
            
            def refresh_loop():
                while True:
                    time.sleep(30)
                    try:
                        # This would need to be implemented with Gradio's real-time updates
                        pass
                    except:
                        break
            
            thread = threading.Thread(target=refresh_loop, daemon=True)
            thread.start()
        
        # Start auto-refresh when component loads
        battery_status.load(auto_refresh)
    
    return {
        "battery_status": battery_status,
        "battery_warning": battery_warning,
        "refresh_battery": refresh_battery_status
    }

def create_battery_tab():
    """Create the Battery Management tab"""
    
    battery_manager = get_battery_manager()
    
    with gr.TabItem("Battery"):
        gr.Markdown("""
        ## 🔋 Battery Management
        Monitor and manage battery usage during image generation to prevent unexpected shutdowns.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Current Battery Status
                gr.Markdown("### Current Status")
                
                battery_info = gr.Textbox(
                    label="Battery Information",
                    value=battery_manager.get_battery_info_string(),
                    interactive=False,
                    lines=3
                )
                
                # Real-time monitoring
                monitoring_status = gr.Textbox(
                    label="Monitoring Status",
                    interactive=False,
                    lines=2
                )
                
                # Quick actions
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Status", variant="secondary")
                    test_check_btn = gr.Button("Test Battery Check", variant="secondary")
            
            with gr.Column(scale=1):
                # Configuration
                gr.Markdown("### Configuration")
                
                enabled_checkbox = gr.Checkbox(
                    label="Enable Battery Monitoring",
                    value=battery_manager.get_config()["enabled"],
                    info="Monitor battery during generation"
                )
                
                stop_percentage_slider = gr.Slider(
                    label="Stop Generation Percentage",
                    minimum=5,
                    maximum=50,
                    value=battery_manager.get_config()["stop_percentage"],
                    step=1,
                    info="Stop generation when battery reaches this level"
                )
                
                pause_checkbox = gr.Checkbox(
                    label="Pause on Low Battery",
                    value=battery_manager.get_config()["pause_on_low_battery"],
                    info="Pause generation when battery is low (10% above stop threshold)"
                )
                
                resume_checkbox = gr.Checkbox(
                    label="Resume on Charge",
                    value=battery_manager.get_config()["resume_on_charge"],
                    info="Automatically resume when charging starts"
                )
                
                notification_checkbox = gr.Checkbox(
                    label="Show Notifications",
                    value=battery_manager.get_config()["notification_enabled"],
                    info="Show battery status notifications"
                )
                
                interval_slider = gr.Slider(
                    label="Check Interval (seconds)",
                    minimum=10,
                    maximum=300,
                    value=battery_manager.get_config()["check_interval"],
                    step=10,
                    info="How often to check battery status"
                )
        
        # Status and warnings
        with gr.Row():
            status_display = gr.Textbox(
                label="Status Messages",
                interactive=False,
                lines=3
            )
        
        # Event handlers
        def update_battery_config(enabled, stop_pct, pause_low, resume_charge, notifications, interval):
            try:
                battery_manager.update_config(
                    enabled=enabled,
                    stop_percentage=int(stop_pct),
                    pause_on_low_battery=pause_low,
                    resume_on_charge=resume_charge,
                    notification_enabled=notifications,
                    check_interval=int(interval)
                )
                
                # Update displays
                info = battery_manager.get_battery_info_string()
                status = f"Configuration updated at {time.ctime()}"
                monitoring = f"Monitoring: {'Enabled' if enabled else 'Disabled'}"
                
                return info, monitoring, status
            except Exception as e:
                return battery_manager.get_battery_info_string(), "Error updating config", f"Error: {str(e)}"
        
        def refresh_battery_info():
            try:
                info = battery_manager.get_battery_info_string()
                
                # Get current status
                config = battery_manager.get_config()
                monitoring = f"Monitoring: {'Enabled' if config['enabled'] else 'Disabled'}\n"
                monitoring += f"Stop at: {config['stop_percentage']}%"
                
                # Check for warnings
                status_parts = []
                if battery_manager.should_stop_generation():
                    status_parts.append("⚠️ CRITICAL: Battery critically low - generation will be stopped")
                elif battery_manager.should_pause_generation():
                    status_parts.append("⚠️ WARNING: Battery low - generation may be paused")
                else:
                    status_parts.append("✅ Battery status OK")
                
                status_parts.append(f"Refreshed at {time.ctime()}")
                status = "\n".join(status_parts)
                
                return info, monitoring, status
            except Exception as e:
                return f"Error: {str(e)}", "Error getting status", f"Error refreshing: {str(e)}"
        
        def test_battery_check():
            try:
                from backend.battery_manager import monitor_battery_during_generation
                
                result = monitor_battery_during_generation()
                
                status_parts = [
                    f"Battery Test Results:",
                    f"Should Stop: {result['should_stop']}",
                    f"Should Pause: {result['should_pause']}",
                    f"Can Resume: {result['can_resume']}",
                    f"Battery Info: {result['battery_info']}",
                    f"Tested at {time.ctime()}"
                ]
                
                return "\n".join(status_parts)
            except Exception as e:
                return f"Error testing battery: {str(e)}"
        
        # Wire up events
        for component in [enabled_checkbox, stop_percentage_slider, pause_checkbox, resume_checkbox, notification_checkbox, interval_slider]:
            component.change(
                update_battery_config,
                inputs=[enabled_checkbox, stop_percentage_slider, pause_checkbox, resume_checkbox, notification_checkbox, interval_slider],
                outputs=[battery_info, monitoring_status, status_display]
            )
        
        refresh_btn.click(
            refresh_battery_info,
            outputs=[battery_info, monitoring_status, status_display]
        )
        
        test_check_btn.click(
            test_battery_check,
            outputs=[status_display]
        )
