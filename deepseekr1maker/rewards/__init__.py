# deepseekr1maker/rewards/__init__.py
from .accuracy_reward import accuracy_reward
from .format_reward import format_reward
from .reasoning_reward import reasoning_steps_reward
from .cosine_reward import get_cosine_scaled_reward
from .repetition_reward import get_repetition_penalty_reward
from .language_reward import language_consistency_reward

def get_reward_functions(reward_config):
    """
    Returns a list of reward functions based on the configuration.
    
    Args:
        reward_config: Reward configuration
    
    Returns:
        List of reward functions
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=reward_config.cosine_min_value_wrong,
            max_value_wrong=reward_config.cosine_max_value_wrong,
            min_value_correct=reward_config.cosine_min_value_correct,
            max_value_correct=reward_config.cosine_max_value_correct,
            max_len=reward_config.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=reward_config.repetition_n_grams,
            max_penalty=reward_config.repetition_max_penalty,
        ),
        "language_consistency": language_consistency_reward,
    }

    for func_name in reward_config.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list

def apply_reward_functions(reward_funcs, completions, **kwargs):
    """
    Apply a list of reward functions to model completions.
    
    Args:
        reward_funcs: List of reward functions to apply
        completions: List of model completions
        **kwargs: Additional arguments to pass to reward functions
        
    Returns:
        Dictionary mapping reward function name to reward values
    """
    rewards = {}
    
    for i, func in enumerate(reward_funcs):
        # Get function name (without module prefix)
        func_name = func.__name__
        if func_name.endswith('_reward'):
            func_name = func_name[:-7]  # Remove '_reward' suffix
            
        # Apply reward function
        try:
            reward_values = func(completions, **kwargs)
            rewards[func_name] = reward_values
        except Exception as e:
            # Log error but continue with other reward functions
            print(f"Error applying reward function {func_name}: {str(e)}")
            rewards[func_name] = [0.0] * len(completions)
    
    return rewards

def combine_rewards(reward_values, reward_weights):
    """
    Combine multiple reward values based on weights.
    
    Args:
        reward_values: Dictionary mapping reward names to lists of reward values
        reward_weights: Dictionary mapping reward names to weight values
        
    Returns:
        List of combined reward values
    """
    if not reward_values:
        return []
    
    # Get number of completions
    n_completions = len(next(iter(reward_values.values())))
    combined_rewards = [0.0] * n_completions
    
    # Calculate weighted sum for each completion
    total_weight = 0.0
    for name, values in reward_values.items():
        weight = reward_weights.get(name, 1.0)
        total_weight += weight
        
        for i in range(n_completions):
            combined_rewards[i] += values[i] * weight
    
    # Normalize by total weight
    if total_weight > 0:
        combined_rewards = [r / total_weight for r in combined_rewards]
    
    return combined_rewards