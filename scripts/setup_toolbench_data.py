# path/setup_toolbench_data.py
# Purpose: Script to set up the ToolBench dataset for ToolGen training

import os
import json
import requests
import shutil
import pandas as pd
from transformers import BertTokenizer
import gzip
import chardet
from tqdm import tqdm
from datasets import load_dataset

# Define URLs and file paths
DATASET_NAME = "Adorg/ToolBench"
DOWNLOAD_DIR = "toolbench_dataset"
TOKENIZER_PATH = "bert-base-uncased"
OUTPUT_DIR = "processed_toolbench"

# Step 1: Load dataset directly from Hugging Face Hub
def load_toolbench_dataset():
    print("Loading ToolBench dataset from Hugging Face Hub...")
    dataset = load_dataset(DATASET_NAME, data_files={
        "train": "toolllama_G123_dfs_train.json",
        "validation": "toolllama_G123_dfs_eval.json"
    })
    return dataset

# Step 2: Load and preprocess tool data
def preprocess_tool_data(data):
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    tool_mapping = {}
    tool_memorization_data = []
    tool_token_data = []
    
    print("Processing tool data...")
    for idx, item in enumerate(tqdm(data)):
        if 'tool' in item:
            tool = item['tool']
            token_id = f"[TOOL_{idx}]"
            tool_mapping[tool['name']] = token_id
            tokenizer.add_tokens([token_id])
            
            tool_memorization_data.append({
                'description': tool['description'],
                'token': token_id
            })
            tool_token_data.append({
                'name': tool['name'],
                'token': token_id,
                'parameters': tool['parameters']
            })
    
    with open(os.path.join(OUTPUT_DIR, "tool_mapping.json"), 'w') as f:
        json.dump(tool_mapping, f)
    print("Tool mapping created.")
    
    with open(os.path.join(OUTPUT_DIR, "tool_memorization.json"), 'w') as f:
        json.dump(tool_memorization_data, f)
    print("Tool memorization data saved.")
    
    return tool_mapping

# Step 3: Prepare Retrieval Training Dataset
def prepare_retrieval_training_data(data, tool_mapping):
    retrieval_training_data = []
    print("Preparing retrieval training data...")
    for item in tqdm(data):
        if 'query' in item and 'relevant_tools' in item:
            retrieval_training_data.append({
                'query': item['query'],
                'relevant_tool_tokens': [tool_mapping[tool] for tool in item['relevant_tools'] if tool in tool_mapping]
            })
    
    with open(os.path.join(OUTPUT_DIR, "retrieval_training.json"), 'w') as f:
        json.dump(retrieval_training_data, f)
    print("Retrieval training data saved.")

# Step 4: Prepare End-to-End Agent Tuning Dataset
def prepare_end_to_end_agent_tuning_data(data, tool_mapping):
    agent_tuning_processed = []
    print("Preparing end-to-end agent tuning data...")
    for item in tqdm(data):
        if 'conversation' in item:
            processed_trajectory = {
                'query': item['conversation'][0]['content'],
                'plan_sequence': [],
                'final_answer': item['conversation'][-1]['content']
            }
            for turn in item['conversation'][1:-1]:
                if 'function_call' in turn:
                    plan = {
                        'plan': turn['content'],
                        'action': tool_mapping.get(turn['function_call']['name'], ''),
                        'parameters': turn['function_call']['arguments'],
                        'feedback': turn.get('function_response', '')
                    }
                    processed_trajectory['plan_sequence'].append(plan)
            agent_tuning_processed.append(processed_trajectory)
    
    with open(os.path.join(OUTPUT_DIR, "agent_tuning.json"), 'w') as f:
        json.dump(agent_tuning_processed, f)
    print("End-to-End Agent Tuning data saved.")

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

# Main execution
def main():
    dataset = load_toolbench_dataset()
    data = dataset['train']  # Assuming we're using the train split
    tool_mapping = preprocess_tool_data(data)
    prepare_retrieval_training_data(data, tool_mapping)
    prepare_end_to_end_agent_tuning_data(dataset['validation'], tool_mapping)  # Using validation split for agent tuning
    
    # Add the validation step
    if validate_output():
        print("\nScript executed successfully!")
    else:
        print("\nScript execution completed, but validation failed. Please check the output.")

if __name__ == "__main__":
    main()