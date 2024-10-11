import json
import os

# Add the following recursive function
def convert_scores(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'score':
                try:
                    obj[key] = float(value)
                except ValueError:
                    print(f"Warning: Unable to convert 'score' to float in entry: {obj}")
                    obj[key] = 0.0  # Assign a default value or handle as needed
            else:
                convert_scores(value)
    elif isinstance(obj, list):
        for item in obj:
            convert_scores(item)

# Modify the existing preprocess_agent_tuning function
def preprocess_agent_tuning(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    convert_scores(data)  # Recursively convert all 'score' fields
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    preprocess_agent_tuning(
        os.path.join(os.path.dirname(__file__), '../data/raw/agent_tuning.json'),  # Relative path
        os.path.join(os.path.dirname(__file__), 'processed_toolbench/agent_tuning.json')
    )