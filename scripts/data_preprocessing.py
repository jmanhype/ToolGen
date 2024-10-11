import json

def preprocess_agent_tuning(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        # Ensure 'score' is a float
        entry['score'] = float(entry['score'])
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    preprocess_agent_tuning(
        '/home/batmanosama/ToolGen/scripts/processed_toolbench/agent_tuning.json',
        '/home/batmanosama/ToolGen/scripts/processed_toolbench/agent_tuning.json'
    )