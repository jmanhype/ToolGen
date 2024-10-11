import json

def validate_agent_tuning(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for entry in data:
            # Example validation: ensure 'score' is a float
            if not isinstance(entry.get('score', 0), (int, float)):
                raise ValueError(f"Invalid type for 'score' in entry: {entry}")
        print("agent_tuning.json validation passed.")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except ValueError as ve:
        print(f"Validation Error: {ve}")

if __name__ == "__main__":
    validate_agent_tuning('/home/batmanosama/ToolGen/scripts/processed_toolbench/agent_tuning.json')