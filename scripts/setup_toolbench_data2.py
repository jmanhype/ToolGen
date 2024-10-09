# scripts/setup_toolbench_data2.py
import os
import json
import re
import ast
from transformers import BertTokenizer
from tqdm import tqdm
from datasets import load_dataset

# Define paths
TOKENIZER_PATH = "bert-base-uncased"
OUTPUT_DIR = "processed_toolbench"
DATASET_NAME = "Adorg/ToolBench"

# Step 1: Load dataset directly from Hugging Face Hub
def load_toolbench_dataset():
    print("Loading ToolBench dataset from Hugging Face Hub...")
    dataset = load_dataset(DATASET_NAME, data_files={
        "train": "toolllama_G123_dfs_train.json",
        "validation": "toolllama_G123_dfs_eval.json"
    })
    return dataset

# Step 2: Load and preprocess tool data
def preprocess_tool_data(dataset):
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    tool_mapping = {}
    tool_memorization_data = []
    tool_token_data = []
    
    print("Processing tool data...")
    for item in tqdm(dataset):
        for conversation in item['conversations']:
            if isinstance(conversation, dict) and conversation.get('from') == 'system':
                tools = extract_tools_from_system_message(conversation['value'])
                for tool in tools:
                    if 'name' in tool and isinstance(tool['name'], str):
                        tool_name = tool['name']
                        if tool_name not in tool_mapping:
                            token_id = f"[TOOL_{len(tool_mapping)}]"
                            tool_mapping[tool_name] = token_id
                            tokenizer.add_tokens([token_id])
                            
                            tool_memorization_data.append({
                                'description': str(tool.get('description', '')),
                                'token': token_id
                            })
                            tool_token_data.append({
                                'name': tool_name,
                                'token': token_id,
                                'parameters': tool.get('parameters', {})
                            })
                break  # We only need to process the system message once per item
    
    with open(os.path.join(OUTPUT_DIR, "tool_mapping.json"), 'w') as f:
        json.dump(tool_mapping, f, indent=2)
    print(f"Tool mapping created with {len(tool_mapping)} tools.")
    
    with open(os.path.join(OUTPUT_DIR, "tool_memorization.json"), 'w') as f:
        json.dump(tool_memorization_data, f, indent=2)
    print(f"Tool memorization data saved with {len(tool_memorization_data)} descriptions.")
    
    return tool_mapping

def extract_tools_from_system_message(message):
    """
    Extract tools from the system message.

    Args:
        message (str): The system message containing the tools.

    Returns:
        list: A list of tools extracted from the message.
    """
    tools = []
    pattern = r"Specifically, you have access to the following APIs:\s*(\[.*?\])\s*$"
    match = re.search(pattern, message, re.DOTALL | re.MULTILINE)
    if match:
        tools_str = match.group(1)
        
        try:
            # Use ast.literal_eval to safely evaluate the string representation of the list
            tools_list = ast.literal_eval(tools_str)
            for tool in tools_list:
                if isinstance(tool, dict) and 'name' in tool:
                    tools.append(tool)
                else:
                    print(f"Skipping invalid tool structure: {tool}")
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing tools list: {e}")
            print(f"Problematic tools string: {tools_str[:200]}...")  # Print first 200 chars
    
    print(f"Extracted {len(tools)} tools")
    for i, tool in enumerate(tools):
        print(f"Tool {i + 1}: {tool.get('name', 'Unknown name')}")
    return tools

# Step 3: Prepare Retrieval Training Dataset
def prepare_retrieval_training_data(dataset, tool_mapping):
    retrieval_training_data = []
    print("Preparing retrieval training data...")
    for item in tqdm(dataset):
        query = None
        relevant_tools = []
        for conversation in item['conversations']:
            if isinstance(conversation, dict):
                if conversation.get('from') == 'user':
                    query = conversation['value']
                elif conversation.get('from') == 'assistant':
                    thought = conversation['value']
                    if 'Action:' in thought:
                        tool = thought.split('Action:')[1].strip().split('\n')[0]
                        if tool in tool_mapping:
                            relevant_tools.append(tool)
        if query and relevant_tools:
            retrieval_training_data.append({
                'query': query,
                'relevant_tool_tokens': [tool_mapping[tool] for tool in relevant_tools]
            })
    
    with open(os.path.join(OUTPUT_DIR, "retrieval_training.json"), 'w') as f:
        json.dump(retrieval_training_data, f)
    print(f"Retrieval training data saved with {len(retrieval_training_data)} examples.")

# Step 4: Prepare End-to-End Agent Tuning Dataset
def prepare_end_to_end_agent_tuning_data(dataset, tool_mapping):
    agent_tuning_processed = []
    print("Preparing end-to-end agent tuning data...")
    for item in tqdm(dataset):
        if not item['conversations'] or not isinstance(item['conversations'][0], dict):
            print(f"Skipping item due to unexpected structure: {item}")
            continue
        
        processed_trajectory = {
            'query': '',
            'plan_sequence': [],
            'final_answer': ''
        }
        
        # Extract query from the first user message
        for conv in item['conversations']:
            if conv.get('from') == 'user':
                processed_trajectory['query'] = conv.get('value', '')
                break
        
        # Extract plan sequence and final answer
        for conv in item['conversations']:
            if conv.get('from') == 'assistant':
                thought = conv.get('value', '')
                if 'Action:' in thought:
                    action = thought.split('Action:')[1].strip().split('\n')[0]
                    plan = {
                        'thought': thought.split('Thought:')[1].strip().split('\n')[0],
                        'action': action,
                        'action_input': {}
                    }
                    if 'Action Input:' in thought:
                        try:
                            action_input = json.loads(thought.split('Action Input:')[1].strip())
                            plan['action_input'] = action_input
                        except json.JSONDecodeError:
                            print(f"Failed to parse action input: {thought}")
                    processed_trajectory['plan_sequence'].append(plan)
            elif conv.get('from') == 'function' and conv.get('name') == 'Finish':
                processed_trajectory['final_answer'] = conv.get('value', '')
        
        agent_tuning_processed.append(processed_trajectory)
    
    with open(os.path.join(OUTPUT_DIR, "agent_tuning.json"), 'w') as f:
        json.dump(agent_tuning_processed, f, indent=2)
    print(f"End-to-End Agent Tuning data saved with {len(agent_tuning_processed)} examples.")

# Add this function to help with debugging
def print_sample_system_message(dataset):
    for item in dataset:
        for conversation in item['conversations']:
            if isinstance(conversation, dict) and conversation.get('from') == 'system':
                print("Sample system message:")
                print(conversation['value'][:500] + "...")  # Print first 500 characters
                return
    print("No system message found in the dataset")

# Add this new function at the end of the file
def validate_output():
    print("\nValidating output...")
    
    # Check if all expected files exist
    expected_files = [
        "tool_mapping.json",
        "tool_memorization.json",
        "retrieval_training.json",
        "agent_tuning.json"
    ]
    for file in expected_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        if not os.path.exists(file_path):
            print(f"Error: {file} not found!")
            return False
        print(f"✓ {file} exists")
    
    # Load and validate content of each file
    with open(os.path.join(OUTPUT_DIR, "tool_mapping.json"), 'r') as f:
        tool_mapping = json.load(f)
        print(f"✓ tool_mapping.json contains {len(tool_mapping)} tools")
    
    with open(os.path.join(OUTPUT_DIR, "tool_memorization.json"), 'r') as f:
        tool_memorization = json.load(f)
        print(f"✓ tool_memorization.json contains {len(tool_memorization)} tool descriptions")
    
    with open(os.path.join(OUTPUT_DIR, "retrieval_training.json"), 'r') as f:
        retrieval_training = json.load(f)
        print(f"✓ retrieval_training.json contains {len(retrieval_training)} training examples")
    
    with open(os.path.join(OUTPUT_DIR, "agent_tuning.json"), 'r') as f:
        agent_tuning = json.load(f)
        print(f"✓ agent_tuning.json contains {len(agent_tuning)} tuning examples")
    
    # Additional checks can be added here
    
    print("\nValidation complete. All files present and contain data.")
    return True

# Modify the main function to include the validation step
def main():
    dataset = load_toolbench_dataset()
    
    print_sample_system_message(dataset['train'])
    
    tool_mapping = preprocess_tool_data(dataset['train'])
    prepare_retrieval_training_data(dataset['train'], tool_mapping)
    
    print("\nSample of validation data:")
    print(json.dumps(dataset['validation'][0], indent=2))
    
    prepare_end_to_end_agent_tuning_data(dataset['validation'], tool_mapping)
    
    if validate_output():
        print("\nScript executed successfully!")
    else:
        print("\nScript execution completed, but validation failed. Please check the output.")

if __name__ == "__main__":
    main()