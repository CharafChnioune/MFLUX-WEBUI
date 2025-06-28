import gradio as gr
from backend.auto_seeds_manager import get_auto_seeds_manager

def create_auto_seeds_tab():
    """Create the Auto Seeds management tab"""
    
    auto_seeds_manager = get_auto_seeds_manager()
    
    with gr.TabItem("Auto Seeds"):
        gr.Markdown("""
        ## 🎲 Auto Seeds Manager
        Manage automatic seed generation and rotation for consistent yet varied results.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Current Configuration
                gr.Markdown("### Configuration")
                
                with gr.Row():
                    enabled_checkbox = gr.Checkbox(
                        label="Enable Auto Seeds",
                        value=auto_seeds_manager.get_config()["enabled"],
                        info="Automatically use seeds from the pool"
                    )
                    
                    shuffle_checkbox = gr.Checkbox(
                        label="Shuffle Mode",
                        value=auto_seeds_manager.get_config()["shuffle"],
                        info="Random selection vs sequential"
                    )
                
                # Pool Management
                gr.Markdown("### Seed Pool Management")
                
                with gr.Row():
                    pool_size_number = gr.Number(
                        label="Generate Pool Size",
                        value=100,
                        minimum=10,
                        maximum=1000,
                        precision=0
                    )
                    
                    generate_pool_btn = gr.Button(
                        "Generate Random Pool",
                        variant="primary"
                    )
                
                with gr.Row():
                    min_seed = gr.Number(
                        label="Min Seed",
                        value=1,
                        precision=0
                    )
                    
                    max_seed = gr.Number(
                        label="Max Seed",
                        value=2**32-1,
                        precision=0
                    )
                
                # Manual Seed Management
                gr.Markdown("### Manual Seed Management")
                
                seeds_textbox = gr.Textbox(
                    label="Seeds (comma separated)",
                    placeholder="12345, 67890, 54321",
                    lines=3,
                    info="Enter seeds separated by commas"
                )
                
                with gr.Row():
                    add_seeds_btn = gr.Button("Add to Pool")
                    remove_seeds_btn = gr.Button("Remove from Pool")
                    exclude_seeds_btn = gr.Button("Exclude Seeds", variant="stop")
                
                # Pool Actions
                gr.Markdown("### Pool Actions")
                
                with gr.Row():
                    clear_pool_btn = gr.Button("Clear Pool", variant="stop")
                    reset_index_btn = gr.Button("Reset Index")
                    refresh_info_btn = gr.Button("Refresh Info", variant="secondary")
            
            with gr.Column(scale=1):
                # Current Status
                gr.Markdown("### Current Status")
                
                pool_info = gr.Textbox(
                    label="Pool Information",
                    value=f"Pool Size: {auto_seeds_manager.get_pool_size()}\nEnabled: {auto_seeds_manager.get_config()['enabled']}\nShuffle: {auto_seeds_manager.get_config()['shuffle']}",
                    lines=5,
                    interactive=False
                )
                
                # Current Seeds Preview
                current_seeds_display = gr.Textbox(
                    label="Current Seeds in Pool (first 20)",
                    value=", ".join(map(str, auto_seeds_manager.get_config()["seed_pool"][:20])),
                    lines=10,
                    interactive=False
                )
                
                # Excluded Seeds
                excluded_seeds_display = gr.Textbox(
                    label="Excluded Seeds",
                    value=", ".join(map(str, auto_seeds_manager.get_config()["exclude_seeds"][:20])),
                    lines=5,
                    interactive=False
                )
                
                # Test Section
                gr.Markdown("### Test Auto Seeds")
                
                test_btn = gr.Button("Get Next Seed", variant="secondary")
                test_result = gr.Textbox(
                    label="Test Result",
                    interactive=False
                )
        
        # Status messages
        status_message = gr.Textbox(
            label="Status",
            interactive=False,
            visible=False
        )
        
        # Event handlers
        def update_enabled(enabled):
            auto_seeds_manager.enable_auto_seeds(enabled)
            return update_info_displays()
        
        def update_shuffle(shuffle):
            auto_seeds_manager.set_shuffle_mode(shuffle)
            return update_info_displays()
        
        def generate_pool(count, min_val, max_val):
            try:
                seeds = auto_seeds_manager.generate_seed_pool(int(count), int(min_val), int(max_val))
                auto_seeds_manager.add_seeds_to_pool(seeds)
                info = update_info_displays()
                return info + (f"Generated {len(seeds)} seeds and added to pool",)
            except Exception as e:
                info = update_info_displays()
                return info + (f"Error generating pool: {str(e)}",)
        
        def parse_seeds(seeds_text):
            try:
                seeds = []
                for seed_str in seeds_text.split(','):
                    seed_str = seed_str.strip()
                    if seed_str:
                        seeds.append(int(seed_str))
                return seeds
            except ValueError:
                return []
        
        def add_seeds(seeds_text):
            try:
                seeds = parse_seeds(seeds_text)
                if not seeds:
                    info = update_info_displays()
                    return info + ("Error: No valid seeds found",)
                
                auto_seeds_manager.add_seeds_to_pool(seeds)
                info = update_info_displays()
                return info + (f"Added {len(seeds)} seeds to pool",)
            except Exception as e:
                info = update_info_displays()
                return info + (f"Error adding seeds: {str(e)}",)
        
        def remove_seeds(seeds_text):
            try:
                seeds = parse_seeds(seeds_text)
                if not seeds:
                    info = update_info_displays()
                    return info + ("Error: No valid seeds found",)
                
                auto_seeds_manager.remove_seeds_from_pool(seeds)
                info = update_info_displays()
                return info + (f"Removed {len(seeds)} seeds from pool",)
            except Exception as e:
                info = update_info_displays()
                return info + (f"Error removing seeds: {str(e)}",)
        
        def exclude_seeds(seeds_text):
            try:
                seeds = parse_seeds(seeds_text)
                if not seeds:
                    info = update_info_displays()
                    return info + ("Error: No valid seeds found",)
                
                auto_seeds_manager.exclude_seeds(seeds)
                info = update_info_displays()
                return info + (f"Excluded {len(seeds)} seeds",)
            except Exception as e:
                info = update_info_displays()
                return info + (f"Error excluding seeds: {str(e)}",)
        
        def clear_pool():
            try:
                auto_seeds_manager.clear_pool()
                info = update_info_displays()
                return info + ("Pool cleared successfully",)
            except Exception as e:
                info = update_info_displays()
                return info + (f"Error clearing pool: {str(e)}",)
        
        def reset_index():
            try:
                auto_seeds_manager.reset_index()
                info = update_info_displays()
                return info + ("Index reset successfully",)
            except Exception as e:
                info = update_info_displays()
                return info + (f"Error resetting index: {str(e)}",)
        
        def update_info_displays():
            config = auto_seeds_manager.get_config()
            pool_size = auto_seeds_manager.get_pool_size()
            
            pool_info_text = f"""Pool Size: {pool_size}
Enabled: {config['enabled']}
Shuffle: {config['shuffle']}
Current Index: {config['current_index']}
Excluded Count: {len(config['exclude_seeds'])}"""
            
            current_seeds_text = ", ".join(map(str, config["seed_pool"][:20]))
            if len(config["seed_pool"]) > 20:
                current_seeds_text += f" ... ({pool_size - 20} more)"
            
            excluded_seeds_text = ", ".join(map(str, config["exclude_seeds"][:20]))
            if len(config["exclude_seeds"]) > 20:
                excluded_seeds_text += f" ... ({len(config['exclude_seeds']) - 20} more)"
            
            return (
                pool_info_text,
                current_seeds_text,
                excluded_seeds_text
            )
        
        def test_next_seed():
            try:
                next_seed = auto_seeds_manager.get_next_seed()
                if next_seed is not None:
                    return f"Next seed: {next_seed}"
                else:
                    return "Auto seeds disabled or pool empty"
            except Exception as e:
                return f"Error getting next seed: {str(e)}"
        
        # Wire up event handlers
        enabled_checkbox.change(
            update_enabled,
            inputs=[enabled_checkbox],
            outputs=[pool_info, current_seeds_display, excluded_seeds_display]
        )
        
        shuffle_checkbox.change(
            update_shuffle,
            inputs=[shuffle_checkbox],
            outputs=[pool_info, current_seeds_display, excluded_seeds_display]
        )
        
        generate_pool_btn.click(
            generate_pool,
            inputs=[pool_size_number, min_seed, max_seed],
            outputs=[pool_info, current_seeds_display, excluded_seeds_display, status_message]
        )
        
        add_seeds_btn.click(
            add_seeds,
            inputs=[seeds_textbox],
            outputs=[pool_info, current_seeds_display, excluded_seeds_display, status_message]
        )
        
        remove_seeds_btn.click(
            remove_seeds,
            inputs=[seeds_textbox],
            outputs=[pool_info, current_seeds_display, excluded_seeds_display, status_message]
        )
        
        exclude_seeds_btn.click(
            exclude_seeds,
            inputs=[seeds_textbox],
            outputs=[pool_info, current_seeds_display, excluded_seeds_display, status_message]
        )
        
        clear_pool_btn.click(
            clear_pool,
            outputs=[pool_info, current_seeds_display, excluded_seeds_display, status_message]
        )
        
        reset_index_btn.click(
            reset_index,
            outputs=[pool_info, current_seeds_display, excluded_seeds_display, status_message]
        )
        
        refresh_info_btn.click(
            update_info_displays,
            outputs=[pool_info, current_seeds_display, excluded_seeds_display]
        )
        
        test_btn.click(
            test_next_seed,
            outputs=[test_result]
        )
        
        # Show status message when it has content
        status_message.change(
            lambda x: gr.update(visible=bool(x.strip())) if x else gr.update(visible=False),
            inputs=[status_message],
            outputs=[status_message]
        )
