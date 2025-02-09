import os
import json
def get_data(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None
file_dir = "./agents/behaviour_cloning_agent/replays/41862933_58818459.json"
replay=get_data(file_dir)
print(type(replay))
print(replay.keys())
print(f"Name:{replay["info"]}")
print(replay["steps"][2][0]["action"])
