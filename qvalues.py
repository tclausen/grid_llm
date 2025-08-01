import random
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List

qvalues = {}

alpha = 0.5
gamma = 0.9

def add(s, s1, a, v, actions):
    td = alpha*(v+gamma*maxQ(s1, actions))
    #print(f"td = {td}, alpha = {alpha}, gamma = {gamma}, maxQ = {maxQ(s1, actions)}, {s}, {s1}")
    if (s, a) not in qvalues:
        qvalues[(s, a)] = td
    else:
        qvalues[(s, a)] = (1-alpha)*qvalues[(s, a)] + td

def maxQ(s, actions):
    maxQ = -100000000000000
    for a in actions:
        if (s, a) not in qvalues: 
            maxQ = max(maxQ, 0) # Unexplored actions have reward 0 by default
            continue
        if qvalues[(s, a)] > maxQ:
            maxQ = qvalues[(s, a)]
    return maxQ

def get(s, a):
    if (s, a) not in qvalues:
        return 0
    return qvalues[(s, a)]

def getBestAction(s, actions):
    bestAction = None
    bestValue = -100000000000000
    unExplored = []
    for a in actions:
        if (s, a) not in qvalues:
            #print(f"{s} {a} unexplored")
            unExplored.append(a)
            continue
        v = qvalues[(s, a)]
        if v > bestValue:
            bestValue = v
            bestAction = a
    # Unexplored actions have a 0 reward by default
    if bestValue <= 0 and len(unExplored) > 0:
        bestAction = random.choice(unExplored)
    return bestAction

def testAdd1():
    assert(qvalues == {})
    add("s1", "s2", "a1", 1, ["a1", "a2"])
    add("s2", "s3", "a2", 1, ["a1", "a2"])
    print(qvalues)
    add("s1", "s2", "a1", 1, ["a1", "a2"])
    print(qvalues)
    assert(get("s1", "a1") == 0.975)
    assert(get("s1", "a2") == 0)
    assert(get("s1", "a3") == 0)
    print(qvalues)
    print(f"Best action in s1: {getBestAction("s1", ["a1", "a2"])}")
    print(f"Best action in s2: {getBestAction("s2", ["a1", "a2"])}")
    print(f"Best action in s3 (random): {getBestAction("s3", ["a1", "a2"])}") 
    assert(getBestAction("s1", ["a1", "a2"]) == "a1")
    assert(getBestAction("s2", ["a1", "a2"]) == "a2")
    print(qvalues)

# ============================================================================
# Q-VALUES CHECKPOINTING SYSTEM
# ============================================================================

def save_qvalues_pickle(filepath: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save Q-values to a pickle file with metadata.
    
    Args:
        filepath (str): Path to save the checkpoint file
        metadata (dict, optional): Additional metadata to save with Q-values
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'qvalues': qvalues.copy(),
            'alpha': alpha,
            'gamma': gamma,
            'timestamp': datetime.now().isoformat(),
            'num_entries': len(qvalues),
            'metadata': metadata or {}
        }
        
        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        print(f"Successfully saved Q-values checkpoint to {filepath}")
        print(f"Saved {len(qvalues)} Q-value entries")
        return True
        
    except Exception as e:
        print(f"Error saving Q-values to {filepath}: {e}")
        return False

def load_qvalues_pickle(filepath: str, replace_current: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load Q-values from a pickle file.
    
    Args:
        filepath (str): Path to the checkpoint file
        replace_current (bool): If True, replace current Q-values; if False, merge
        
    Returns:
        dict: Checkpoint metadata if successful, None otherwise
    """
    global qvalues, alpha, gamma
    
    try:
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return None
            
        # Load from pickle file
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Extract data
        loaded_qvalues = checkpoint_data.get('qvalues', {})
        loaded_alpha = checkpoint_data.get('alpha', alpha)
        loaded_gamma = checkpoint_data.get('gamma', gamma)
        metadata = checkpoint_data.get('metadata', {})
        
        # Update Q-values
        if replace_current:
            qvalues.clear()
            qvalues.update(loaded_qvalues)
        else:
            # Merge Q-values (loaded values override existing ones)
            qvalues.update(loaded_qvalues)
        
        # Update parameters
        alpha = loaded_alpha
        gamma = loaded_gamma
        
        print(f"Successfully loaded Q-values checkpoint from {filepath}")
        print(f"Loaded {len(loaded_qvalues)} Q-value entries")
        print(f"Alpha: {alpha}, Gamma: {gamma}")
        
        return {
            'timestamp': checkpoint_data.get('timestamp'),
            'num_entries': checkpoint_data.get('num_entries'),
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error loading Q-values from {filepath}: {e}")
        return None

def save_qvalues_json(filepath: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save Q-values to a JSON file (human-readable format).
    Note: JSON keys must be strings, so state-action pairs are converted to strings.
    
    Args:
        filepath (str): Path to save the checkpoint file
        metadata (dict, optional): Additional metadata to save with Q-values
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert Q-values to JSON-serializable format
        json_qvalues = {}
        for (state, action), value in qvalues.items():
            # Create a string key for the state-action pair
            key = f"{state}|{action}"
            json_qvalues[key] = value
        
        # Prepare checkpoint data
        checkpoint_data = {
            'qvalues': json_qvalues,
            'alpha': alpha,
            'gamma': gamma,
            'timestamp': datetime.now().isoformat(),
            'num_entries': len(qvalues),
            'metadata': metadata or {}
        }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        print(f"Successfully saved Q-values checkpoint to {filepath}")
        print(f"Saved {len(qvalues)} Q-value entries (JSON format)")
        return True
        
    except Exception as e:
        print(f"Error saving Q-values to {filepath}: {e}")
        return False

def load_qvalues_json(filepath: str, replace_current: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load Q-values from a JSON file.
    
    Args:
        filepath (str): Path to the checkpoint file
        replace_current (bool): If True, replace current Q-values; if False, merge
        
    Returns:
        dict: Checkpoint metadata if successful, None otherwise
    """
    global qvalues, alpha, gamma
    
    try:
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return None
            
        # Load from JSON file
        with open(filepath, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Extract and convert Q-values back to the original format
        json_qvalues = checkpoint_data.get('qvalues', {})
        loaded_qvalues = {}
        
        for key, value in json_qvalues.items():
            # Parse the string key back to (state, action) tuple
            if '|' in key:
                parts = key.split('|', 1)  # Split only on first '|' in case state contains '|'
                state, action = parts[0], parts[1]
                loaded_qvalues[(state, action)] = value
        
        loaded_alpha = checkpoint_data.get('alpha', alpha)
        loaded_gamma = checkpoint_data.get('gamma', gamma)
        metadata = checkpoint_data.get('metadata', {})
        
        # Update Q-values
        if replace_current:
            qvalues.clear()
            qvalues.update(loaded_qvalues)
        else:
            # Merge Q-values (loaded values override existing ones)
            qvalues.update(loaded_qvalues)
        
        # Update parameters
        alpha = loaded_alpha
        gamma = loaded_gamma
        
        print(f"Successfully loaded Q-values checkpoint from {filepath}")
        print(f"Loaded {len(loaded_qvalues)} Q-value entries")
        print(f"Alpha: {alpha}, Gamma: {gamma}")
        
        return {
            'timestamp': checkpoint_data.get('timestamp'),
            'num_entries': checkpoint_data.get('num_entries'),
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error loading Q-values from {filepath}: {e}")
        return None

def save_qvalues(filepath: str, format: str = 'pickle', metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save Q-values in the specified format.
    
    Args:
        filepath (str): Path to save the checkpoint file
        format (str): Format to save ('pickle' or 'json')
        metadata (dict, optional): Additional metadata to save with Q-values
        
    Returns:
        bool: True if successful, False otherwise
    """
    if format.lower() == 'pickle':
        return save_qvalues_pickle(filepath, metadata)
    elif format.lower() == 'json':
        return save_qvalues_json(filepath, metadata)
    else:
        print(f"Unsupported format: {format}. Use 'pickle' or 'json'.")
        return False

def load_qvalues(filepath: str, replace_current: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load Q-values from a checkpoint file (auto-detects format).
    
    Args:
        filepath (str): Path to the checkpoint file
        replace_current (bool): If True, replace current Q-values; if False, merge
        
    Returns:
        dict: Checkpoint metadata if successful, None otherwise
    """
    if filepath.endswith('.json'):
        return load_qvalues_json(filepath, replace_current)
    elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        return load_qvalues_pickle(filepath, replace_current)
    else:
        # Try pickle first, then JSON
        result = load_qvalues_pickle(filepath, replace_current)
        if result is None:
            result = load_qvalues_json(filepath, replace_current)
        return result

def clear_qvalues():
    """Clear all Q-values."""
    global qvalues
    qvalues.clear()
    print("Cleared all Q-values")

def get_qvalues_summary() -> Dict[str, Any]:
    """
    Get a summary of current Q-values.
    
    Returns:
        dict: Summary information about Q-values
    """
    if not qvalues:
        return {
            'num_entries': 0,
            'states': 0,
            'actions': 0,
            'value_range': (0, 0),
            'parameters': {'alpha': alpha, 'gamma': gamma}
        }
    
    # Extract states and actions
    states = set()
    actions = set()
    values = []
    
    for (state, action), value in qvalues.items():
        states.add(state)
        actions.add(action)
        values.append(value)
    
    return {
        'num_entries': len(qvalues),
        'states': len(states),
        'actions': len(actions),
        'value_range': (min(values), max(values)),
        'avg_value': sum(values) / len(values),
        'parameters': {'alpha': alpha, 'gamma': gamma}
    }

def list_checkpoints(checkpoint_dir: str = 'qvalue_checkpoints') -> List[Dict[str, Any]]:
    """
    List available Q-value checkpoint files.
    
    Args:
        checkpoint_dir (str): Directory to search for checkpoints
        
    Returns:
        list: List of checkpoint information
    """
    checkpoints = []
    
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for filename in os.listdir(checkpoint_dir):
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(filepath) and (filename.endswith('.pkl') or filename.endswith('.pickle') or filename.endswith('.json')):
            try:
                # Get file stats
                stat = os.stat(filepath)
                size = stat.st_size
                modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                # Try to load metadata
                metadata = None
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        metadata = {
                            'timestamp': data.get('timestamp'),
                            'num_entries': data.get('num_entries'),
                            'alpha': data.get('alpha'),
                            'gamma': data.get('gamma'),
                            'metadata': data.get('metadata', {})
                        }
                    except:
                        pass
                elif filename.endswith('.pkl') or filename.endswith('.pickle'):
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        metadata = {
                            'timestamp': data.get('timestamp'),
                            'num_entries': data.get('num_entries'),
                            'alpha': data.get('alpha'),
                            'gamma': data.get('gamma'),
                            'metadata': data.get('metadata', {})
                        }
                    except:
                        pass
                
                checkpoints.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size_bytes': size,
                    'modified': modified,
                    'metadata': metadata
                })
                
            except Exception as e:
                print(f"Error reading checkpoint {filename}: {e}")
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    return checkpoints

def create_checkpoint_name(prefix: str = 'qvalues', suffix: str = '', 
                          include_timestamp: bool = True, format: str = 'pickle') -> str:
    """
    Create a standardized checkpoint filename.
    
    Args:
        prefix (str): Filename prefix
        suffix (str): Filename suffix
        include_timestamp (bool): Whether to include timestamp
        format (str): File format ('pickle' or 'json')
        
    Returns:
        str: Generated filename
    """
    parts = [prefix]
    
    if suffix:
        parts.append(suffix)
    
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts.append(timestamp)
    
    filename = '_'.join(parts)
    
    if format.lower() == 'json':
        filename += '.json'
    else:
        filename += '.pkl'
    
    return filename

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_checkpointing():
    """Test the Q-value checkpointing system."""
    print("=== Testing Q-value Checkpointing System ===")
    
    # Clear existing Q-values
    clear_qvalues()
    
    # Add some test Q-values
    print("\n1. Adding test Q-values...")
    add("s1", "s2", "u", 1.0, ["u", "d", "l", "r"])
    add("s1", "s2", "d", 0.5, ["u", "d", "l", "r"])
    add("s2", "s3", "u", 1.5, ["u", "d", "l", "r"])
    
    print(f"Current Q-values: {qvalues}")
    print(f"Summary: {get_qvalues_summary()}")
    
    # Test pickle save/load
    print("\n2. Testing pickle format...")
    checkpoint_dir = "test_qvalue_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    pickle_file = os.path.join(checkpoint_dir, "test_qvalues.pkl")
    metadata = {"test_run": "checkpointing_test", "policy": "epsilon_greedy"}
    
    # Save
    save_result = save_qvalues_pickle(pickle_file, metadata)
    print(f"Save result: {save_result}")
    
    # Clear and load
    original_qvalues = qvalues.copy()
    clear_qvalues()
    print(f"After clearing: {len(qvalues)} entries")
    
    load_result = load_qvalues_pickle(pickle_file)
    print(f"Load result: {load_result}")
    print(f"After loading: {len(qvalues)} entries")
    print(f"Q-values match: {qvalues == original_qvalues}")
    
    # Test JSON save/load
    print("\n3. Testing JSON format...")
    json_file = os.path.join(checkpoint_dir, "test_qvalues.json")
    
    # Save
    save_result = save_qvalues_json(json_file, metadata)
    print(f"Save result: {save_result}")
    
    # Clear and load
    clear_qvalues()
    print(f"After clearing: {len(qvalues)} entries")
    
    load_result = load_qvalues_json(json_file)
    print(f"Load result: {load_result}")
    print(f"After loading: {len(qvalues)} entries")
    print(f"Q-values match: {qvalues == original_qvalues}")
    
    # Test checkpoint listing
    print("\n4. Testing checkpoint listing...")
    checkpoints = list_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp['filename']}: {cp['metadata']['num_entries']} entries, {cp['size_bytes']} bytes")
    
    # Test auto-format detection
    print("\n5. Testing auto-format detection...")
    clear_qvalues()
    load_result = load_qvalues(pickle_file)  # Should auto-detect pickle
    print(f"Auto-detect pickle: {load_result is not None}")
    
    clear_qvalues()
    load_result = load_qvalues(json_file)  # Should auto-detect JSON
    print(f"Auto-detect JSON: {load_result is not None}")
    
    # Cleanup
    try:
        os.remove(pickle_file)
        os.remove(json_file)
        os.rmdir(checkpoint_dir)
        print("\nCleanup completed.")
    except:
        print("\nCleanup failed (files may still exist).")
    
    print("\n=== Checkpointing test completed ===")

if __name__ == "__main__":
    # Run original tests
    testAdd1()
    
    # Run checkpointing tests
    test_checkpointing()