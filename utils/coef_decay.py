import torch
import numpy as np

def update_entropy_coef(min_entropy_coef, initial_entropy_coef, current_step, decay_steps, decay_type="linear"):
    """Decay entropy coefficient over time."""
    if decay_type == "linear":
        entropy_coef = max(
            min_entropy_coef, 
            initial_entropy_coef * (1 - current_step / decay_steps)
        )
    
    elif decay_type == "exponential":
        decay_rate = 0.99  # Adjust as needed
        entropy_coef = max(
            min_entropy_coef, 
            initial_entropy_coef * (decay_rate ** (current_step / 10000))
        )


    return entropy_coef