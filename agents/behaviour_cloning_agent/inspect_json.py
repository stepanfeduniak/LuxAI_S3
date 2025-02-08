import os
import json
def get_data(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None
file_dir = "./agents/behaviour_cloning_agent/41863713_58807861.json"
replay=get_data(file_dir)
print(type(replay))
print(replay.keys())
print(replay["steps"][2][0]["action"])
