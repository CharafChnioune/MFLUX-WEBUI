import gc
import mlx.core as mx

def force_mlx_cleanup():
    """Clean up MLX memory and resources"""
    mx.eval(mx.zeros(1))
    
    # mlx.core deprecated mx.metal.* in favor of top-level helpers.
    clear_cache = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if clear_cache:
        clear_cache()
    
    reset_peak = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if reset_peak:
        reset_peak()

    gc.collect()

def print_memory_usage(label):
    """Print current MLX memory usage"""
    try:
        # mlx.core deprecated mx.metal.get_*_memory in favor of mx.get_*_memory.
        get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        active_memory = get_active() / 1e6
        peak_memory = get_peak() / 1e6
        print(f"{label} - Active memory: {active_memory:.2f} MB, Peak memory: {peak_memory:.2f} MB")
    except Exception as e:
        print(f"Error getting memory usage: {str(e)}") 
