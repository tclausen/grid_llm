#!/usr/bin/env python3

"""
Test script for training infrastructure
"""

import torch
import os
import tempfile
from train import GridWorldTrainer, get_default_config
from evaluate import GridWorldEvaluator
from log import getLog

log = getLog()

def test_training_infrastructure():
    """Test the complete training infrastructure"""
    
    print("=== Testing Training Infrastructure ===")
    
    # Create a minimal config for testing
    config = {
        'block_size': 64,
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 128,
        'dropout': 0.1,
        
        'max_epochs': 3,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        'scheduler': 'cosine',
        'scheduler_step_size': 2,
        'scheduler_gamma': 0.5,
        
        'num_traces': 20,
        'steps_per_trace': 8,
        'policy': 'random',
        'train_ratio': 0.8,
        
        'log_interval': 1,
        'save_interval': 1,
        'output_dir': 'test_checkpoints',
    }
    
    # Clean up any existing test directory
    if os.path.exists(config['output_dir']):
        import shutil
        shutil.rmtree(config['output_dir'])
    
    try:
        # Test training
        print("\n1. Testing trainer initialization...")
        trainer = GridWorldTrainer(config)
        print(f"✓ Trainer created with device: {trainer.device}")
        print(f"✓ Model parameters: {trainer.model.get_num_params():,}")
        
        print("\n2. Testing data generation...")
        train_loader, val_loader, tokenized_data = trainer.generate_training_data()
        print(f"✓ Generated {len(tokenized_data['tokenized_sequences'])} tokenized sequences")
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader) if val_loader else 0}")
        
        print("\n3. Testing single batch training...")
        # Test single batch
        batch = next(iter(train_loader))
        loss = trainer.compute_loss(batch)
        metrics = trainer.compute_metrics(batch)
        print(f"✓ Batch loss: {loss.item():.4f}")
        print(f"✓ Batch accuracy: {metrics['accuracy']:.3f}")
        print(f"✓ Action accuracy: {metrics['action_accuracy']:.3f}")
        print(f"✓ State accuracy: {metrics['state_accuracy']:.3f}")
        
        print("\n4. Testing full training loop...")
        model = trainer.train()
        print("✓ Training completed successfully")
        
        # Check that checkpoints were saved
        checkpoint_files = [
            'checkpoint_latest.pt',
            'checkpoint_best.pt',
            'config.json'
        ]
        
        for filename in checkpoint_files:
            filepath = os.path.join(config['output_dir'], filename)
            if os.path.exists(filepath):
                print(f"✓ {filename} saved")
            else:
                print(f"✗ {filename} missing")
        
        print("\n5. Testing model loading and evaluation...")
        # Test loading the trained model
        model_path = os.path.join(config['output_dir'], 'checkpoint_best.pt')
        evaluator = GridWorldEvaluator(model_path)
        print("✓ Model loaded for evaluation")
        
        # Test prediction
        test_state = "***# ****"
        test_action = "r"
        predicted_state, confidence = evaluator.predict_next_state(test_state, test_action)
        print(f"✓ Prediction test: '{test_state}' + '{test_action}' → '{predicted_state}' (conf: {confidence:.3f})")
        
        # Quick evaluation on small test set
        print("\n6. Testing evaluation metrics...")
        from traceGeneration import generateTokenizedTraces
        from world import WorldComplex
        
        world = WorldComplex()
        test_data = generateTokenizedTraces(world, num_traces=10, steps_per_trace=5)
        
        results = evaluator.evaluate_state_transitions(test_data)
        print(f"✓ Evaluated {results['total_transitions']} transitions")
        print(f"✓ Exact match accuracy: {results['exact_match_accuracy']:.3f}")
        print(f"✓ Character accuracy: {results['character_accuracy']:.3f}")
        
        print("\n=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up test directory
        if os.path.exists(config['output_dir']):
            import shutil
            shutil.rmtree(config['output_dir'])
            print(f"✓ Cleaned up {config['output_dir']}")

def test_model_size_scaling():
    """Test different model sizes to verify scalability"""
    
    print("\n=== Testing Model Size Scaling ===")
    
    configs = [
        {'n_layer': 2, 'n_head': 2, 'n_embd': 64, 'name': 'tiny'},
        {'n_layer': 4, 'n_head': 4, 'n_embd': 128, 'name': 'small'},
        {'n_layer': 6, 'n_head': 6, 'n_embd': 256, 'name': 'medium'},
    ]
    
    base_config = get_default_config()
    base_config['max_epochs'] = 1
    base_config['num_traces'] = 10
    base_config['batch_size'] = 2
    
    for config_override in configs:
        try:
            # Create config
            test_config = base_config.copy()
            test_config.update(config_override)
            test_config['output_dir'] = f"test_{config_override['name']}"
            
            # Create trainer
            trainer = GridWorldTrainer(test_config)
            param_count = trainer.model.get_num_params()
            
            print(f"✓ {config_override['name']}: {param_count:,} parameters")
            
            # Clean up
            if os.path.exists(test_config['output_dir']):
                import shutil
                shutil.rmtree(test_config['output_dir'])
                
        except Exception as e:
            print(f"✗ {config_override['name']} failed: {e}")

if __name__ == "__main__":
    # Run tests
    success = test_training_infrastructure()
    
    if success:
        test_model_size_scaling()
        print("\n🎉 All training infrastructure tests completed successfully!")
    else:
        print("\n❌ Training infrastructure tests failed!")
        exit(1)