"""Tests for the rewards module."""

import pytest
from deepseekr1maker.rewards import get_reward_functions, apply_reward_functions
from deepseekr1maker.rewards.reasoning_reward import reasoning_steps_reward
from deepseekr1maker.rewards.accuracy_reward import accuracy_reward
from deepseekr1maker.config.reward_config import RewardConfig

def test_get_reward_functions():
    """Test get_reward_functions with different configurations."""
    # Test with default config
    config = RewardConfig()
    reward_funcs = get_reward_functions(config)
    assert len(reward_funcs) > 0
    
    # Test with custom config
    custom_config = RewardConfig(reward_funcs=["accuracy", "reasoning_steps"])
    reward_funcs = get_reward_functions(custom_config)
    assert len(reward_funcs) == 2

def test_apply_reward_functions():
    """Test apply_reward_functions with sample completions."""
    # Create sample completions
    completions = [
        [{"content": "Step 1: Understand the problem\nStep 2: Solve it\nStep 3: Verify the solution"}],
        [{"content": "The answer is 42."}]
    ]
    
    # Create sample reward functions
    reward_funcs = [
        reasoning_steps_reward,
        accuracy_reward
    ]
    
    # Apply reward functions
    rewards = apply_reward_functions(reward_funcs, completions)
    
    # Check that rewards were calculated
    assert len(rewards) == len(completions)
    assert all(isinstance(r, (int, float)) for r in rewards)

def test_reasoning_steps_reward():
    """Test reasoning_steps_reward function."""
    # Test with good reasoning steps
    good_completion = [
        [{"content": "Step 1: Understand the problem\nStep 2: Solve it\nStep 3: Verify the solution"}]
    ]
    good_reward = reasoning_steps_reward(good_completion)
    assert good_reward[0] > 0.5  # Should have a high reward
    
    # Test with poor reasoning steps
    poor_completion = [
        [{"content": "The answer is 42."}]
    ]
    poor_reward = reasoning_steps_reward(poor_completion)
    assert poor_reward[0] < 0.5  # Should have a lower reward
    
    # Test with diagnostics
    diagnostics = reasoning_steps_reward(good_completion, return_diagnostics=True)
    assert isinstance(diagnostics[0], dict)
    assert "score" in diagnostics[0]
    assert "steps_found" in diagnostics[0]

def test_accuracy_reward():
    """Test accuracy_reward function."""
    # Test with correct answer
    correct_completion = [
        [{"content": "The answer is 42."}]
    ]
    reference = ["The answer is 42."]
    correct_reward = accuracy_reward(correct_completion, reference=reference)
    assert correct_reward[0] > 0.5  # Should have a high reward
    
    # Test with incorrect answer
    incorrect_completion = [
        [{"content": "The answer is 24."}]
    ]
    incorrect_reward = accuracy_reward(incorrect_completion, reference=reference)
    assert incorrect_reward[0] < 0.5  # Should have a lower reward 