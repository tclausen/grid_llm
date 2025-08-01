#!/usr/bin/env python3

"""
Evaluation utilities for Grid World LLM
"""

import torch
import torch.nn.functional as F
import os
import json
import argparse
from collections import defaultdict
import numpy as np

from model.model import GPT, GPTConfig
from tokenizer import CharTokenizer
from dataset import create_train_val_datasets, create_dataloaders
from traceGeneration import generateTokenizedTraces
from world import WorldComplex
from state import State
from log import getLog

log = getLog()

class GridWorldEvaluator:
    """Evaluator for Grid World LLM"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize evaluator
        
        Args:
            model_path (str): Path to trained model checkpoint
            device (str, optional): Device to use
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = CharTokenizer()
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Recreate model
        model_config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=self.config['block_size'],
            n_layer=self.config['n_layer'],
            n_head=self.config['n_head'],
            n_embd=self.config['n_embd'],
            dropout=0.0  # No dropout during evaluation
        )
        
        self.model = GPT.from_scratch(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        log.info(f"Loaded model from {model_path}")
        log.info(f"Model config: {model_config}")
    
    def predict_next_state(self, state_representation, action):
        """
        Predict next state given current state and action
        
        Args:
            state_representation (str): Current state (9-char string)
            action (str): Action to take ('u', 'd', 'l', 'r')
            
        Returns:
            tuple: (predicted_next_state, confidence_score)
        """
        # Format input as state|action>
        input_text = f"{state_representation}|{action}>"
        
        # Tokenize
        tokens = self.tokenizer.encode_with_special_tokens(input_text, add_bos=True, add_eos=False)
        input_tensor = torch.tensor(tokens, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            
            # Get logits for next token prediction
            next_token_logits = logits[0, -1, :]  # Last token's predictions
            
            # Get top predictions for each position in the 9-character state
            predicted_state = ""
            total_confidence = 0.0
            
            # Predict 9 characters for the next state
            current_input = input_tensor
            for i in range(9):
                with torch.no_grad():
                    logits, _ = self.model(current_input)
                    next_token_logits = logits[0, -1, :]
                    
                    # Get probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Sample or take most likely
                    next_token_id = torch.argmax(probs).item()
                    confidence = probs[next_token_id].item()
                    
                    # Decode token
                    next_char = self.tokenizer.decode([next_token_id])
                    
                    # Stop if we hit a special token or separator
                    if next_char in ['|', '>', '<BOS>', '<EOS>', '<PAD>']:
                        break
                    
                    predicted_state += next_char
                    total_confidence += confidence
                    
                    # Add token to input for next prediction
                    new_token = torch.tensor([[next_token_id]], device=self.device)
                    current_input = torch.cat([current_input, new_token], dim=1)
                    
                    # Limit sequence length
                    if current_input.size(1) > self.config['block_size'] - 1:
                        current_input = current_input[:, 1:]
            
            avg_confidence = total_confidence / max(len(predicted_state), 1)
            
            # Pad to 9 characters if necessary
            if len(predicted_state) < 9:
                predicted_state += ' ' * (9 - len(predicted_state))
            elif len(predicted_state) > 9:
                predicted_state = predicted_state[:9]
        
        return predicted_state, avg_confidence
    
    def evaluate_state_transitions(self, test_data):
        """
        Evaluate model on state transition prediction
        
        Args:
            test_data (dict): Test data from generateTokenizedTraces
            
        Returns:
            dict: Evaluation results
        """
        log.info("Evaluating state transition predictions...")
        
        results = {
            'total_transitions': 0,
            'correct_predictions': 0,
            'exact_match_accuracy': 0.0,
            'character_accuracy': 0.0,
            'action_breakdown': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'confidence_scores': [],
        }
        
        total_chars_correct = 0
        total_chars = 0
        
        for trace in test_data['traces']:
            llm_formatted = trace.formatForLLM()
            
            for transition in llm_formatted:
                # Parse: state|action>next_state
                if '|' not in transition or '>' not in transition:
                    continue
                
                parts = transition.split('|')
                if len(parts) != 2:
                    continue
                    
                current_state = parts[0]
                action_and_next = parts[1].split('>')
                if len(action_and_next) != 2:
                    continue
                
                action = action_and_next[0]
                true_next_state = action_and_next[1]
                
                # Predict next state
                predicted_next_state, confidence = self.predict_next_state(current_state, action)
                
                # Update results
                results['total_transitions'] += 1
                results['confidence_scores'].append(confidence)
                
                # Exact match
                if predicted_next_state == true_next_state:
                    results['correct_predictions'] += 1
                    results['action_breakdown'][action]['correct'] += 1
                
                results['action_breakdown'][action]['total'] += 1
                
                # Character-level accuracy
                for i in range(min(len(predicted_next_state), len(true_next_state))):
                    if predicted_next_state[i] == true_next_state[i]:
                        total_chars_correct += 1
                    total_chars += 1
        
        # Calculate final metrics
        if results['total_transitions'] > 0:
            results['exact_match_accuracy'] = results['correct_predictions'] / results['total_transitions']
        
        if total_chars > 0:
            results['character_accuracy'] = total_chars_correct / total_chars
        
        results['avg_confidence'] = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0.0
        
        # Per-action accuracy
        for action in results['action_breakdown']:
            action_stats = results['action_breakdown'][action]
            if action_stats['total'] > 0:
                action_stats['accuracy'] = action_stats['correct'] / action_stats['total']
        
        return results
    
    def evaluate_planning_capability(self, world, start_positions, plan_length=5):
        """
        Evaluate model's ability to plan multi-step sequences
        
        Args:
            world: World environment
            start_positions (list): List of starting positions to test
            plan_length (int): Number of steps to plan ahead
            
        Returns:
            dict: Planning evaluation results
        """
        log.info(f"Evaluating planning capability over {plan_length} steps...")
        
        results = {
            'total_plans': 0,
            'successful_plans': 0,
            'planning_accuracy': 0.0,
            'step_accuracies': [0.0] * plan_length,
            'avg_plan_confidence': 0.0
        }
        
        total_confidence = 0.0
        step_correct_counts = [0] * plan_length
        
        for start_pos in start_positions:
            try:
                # Create initial state
                current_state = State([start_pos[0], start_pos[1]])
                current_world_state = world.look(current_state)
                
                # Plan sequence
                planned_states = [current_world_state]
                plan_confidences = []
                
                temp_state = current_state.copy()
                for step in range(plan_length):
                    # Choose a random valid action
                    available_actions = world.actionSpaceInPosition(temp_state.position())
                    if not available_actions:
                        break
                    
                    action = available_actions[0]  # Take first available action
                    
                    # Predict next state with model
                    predicted_next_state, confidence = self.predict_next_state(
                        world.look(temp_state), action
                    )
                    
                    # Get actual next state
                    actual_next_state, reward = world.step(temp_state, action)
                    actual_state_rep = world.look(actual_next_state)
                    
                    # Compare prediction vs reality
                    is_correct = predicted_next_state == actual_state_rep
                    if is_correct:
                        step_correct_counts[step] += 1
                    
                    plan_confidences.append(confidence)
                    temp_state = actual_next_state
                
                results['total_plans'] += 1
                total_confidence += np.mean(plan_confidences) if plan_confidences else 0.0
                
            except Exception as e:
                log.warning(f"Error evaluating plan from {start_pos}: {e}")
                continue
        
        # Calculate final metrics
        if results['total_plans'] > 0:
            results['avg_plan_confidence'] = total_confidence / results['total_plans']
            
            for step in range(plan_length):
                if results['total_plans'] > 0:
                    results['step_accuracies'][step] = step_correct_counts[step] / results['total_plans']
        
        return results
    
    def generate_sample_predictions(self, num_samples=5):
        """Generate and display sample predictions"""
        log.info(f"Generating {num_samples} sample predictions...")
        
        world = WorldComplex()
        
        for i in range(num_samples):
            # Get random position
            pos = world.randomFreePos()
            state = State([pos.x, pos.y])
            current_state_rep = world.look(state)
            
            # Get available actions
            available_actions = world.actionSpaceInPosition(state.position())
            if not available_actions:
                continue
            
            action = available_actions[0]
            
            # Get actual next state
            actual_next_state, reward = world.step(state.copy(), action)
            actual_next_state_rep = world.look(actual_next_state)
            
            # Get model prediction
            predicted_next_state, confidence = self.predict_next_state(current_state_rep, action)
            
            print(f"\nSample {i+1}:")
            print(f"  Current state: '{current_state_rep}'")
            print(f"  Action: '{action}'")
            print(f"  Actual next: '{actual_next_state_rep}'")
            print(f"  Predicted:   '{predicted_next_state}'")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {'✓' if predicted_next_state == actual_next_state_rep else '✗'}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Grid World LLM')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--test_traces', type=int, default=100, help='Number of test traces')
    parser.add_argument('--plan_length', type=int, default=5, help='Planning horizon for evaluation')
    parser.add_argument('--samples', type=int, default=10, help='Number of sample predictions to show')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GridWorldEvaluator(args.model_path)
    
    # Generate test data
    log.info("Generating test data...")
    world = WorldComplex()
    test_data = generateTokenizedTraces(
        world,
        num_traces=args.test_traces,
        steps_per_trace=10,
        policy='random'
    )
    
    # Evaluate state transitions
    transition_results = evaluator.evaluate_state_transitions(test_data)
    
    print(f"\n=== State Transition Evaluation ===")
    print(f"Total transitions: {transition_results['total_transitions']}")
    print(f"Exact match accuracy: {transition_results['exact_match_accuracy']:.3f}")
    print(f"Character accuracy: {transition_results['character_accuracy']:.3f}")
    print(f"Average confidence: {transition_results['avg_confidence']:.3f}")
    
    print(f"\nPer-action accuracy:")
    for action, stats in transition_results['action_breakdown'].items():
        if stats['total'] > 0:
            print(f"  {action}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    
    # Evaluate planning
    start_positions = [[5, 5], [10, 10], [15, 15], [8, 12], [12, 8]]
    planning_results = evaluator.evaluate_planning_capability(world, start_positions, args.plan_length)
    
    print(f"\n=== Planning Capability Evaluation ===")
    print(f"Total plans: {planning_results['total_plans']}")
    print(f"Average confidence: {planning_results['avg_plan_confidence']:.3f}")
    print(f"Step-by-step accuracy:")
    for step, acc in enumerate(planning_results['step_accuracies']):
        print(f"  Step {step+1}: {acc:.3f}")
    
    # Show sample predictions
    print(f"\n=== Sample Predictions ===")
    evaluator.generate_sample_predictions(args.samples)

if __name__ == "__main__":
    main()