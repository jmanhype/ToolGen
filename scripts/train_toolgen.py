# path/finetune_toolgen_model.py
# Purpose: Script to fine-tune the ToolGen model with the ToolBench dataset for tool memorization, retrieval, and agent tuning

import os
import traceback

# **Set GCC and G++ to version 10 to ensure CUDA compatibility**
# Ensure that GCC-10 and G++-10 are installed on your system.
# You can install them via your package manager or using Conda.
# Example using Conda:
# conda install -c conda-forge gcc_linux-64=10 gxx_linux-64=10

# Add these lines before other imports to set the environment variables early
os.environ['CC'] = 'gcc-10'
os.environ['CXX'] = 'g++-10'

# Proceed with the rest of your imports and code
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd  # Add this import statement
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import gc
import psutil
import GPUtil

# Define constants
MODEL_PATH = "unsloth/Llama-3.2-1B"
OUTPUT_DIR = os.path.join("scripts", "processed_toolbench")
FINETUNED_MODEL_DIR = "finetuned_toolgen_model"
LEARNING_RATE = 5e-5
BATCH_SIZE = 1  # Reduced batch size to fit in memory
EPOCHS = 3

# Step 1: Load data for tool memorization, retrieval training, and agent tuning
def load_training_data():
    print("Loading training data...")
    data_files = [
        "tool_memorization.json",
        "retrieval_training.json",
        "agent_tuning.json"
    ]
    datasets = {}
    
    for file_name in data_files:
        file_path = os.path.join(OUTPUT_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping this file.")
            continue
        try:
            # Load JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Handle 'plan_sequence' column if it exists
            if 'plan_sequence' in df.columns:
                df['plan_sequence'] = df['plan_sequence'].apply(json.dumps)
            
            # Convert DataFrame to Dataset
            datasets[file_name] = Dataset.from_pandas(df)
            print(f"Successfully loaded {file_name}. Number of examples: {len(datasets[file_name])}")
            
            # Print the first few rows and data types of the dataset
            print(f"First few rows of {file_name}:")
            print(df.head())
            print(f"\nData types of {file_name}:")
            print(df.dtypes)
            print("\n")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if not datasets:
        raise FileNotFoundError(f"No valid data files found in {OUTPUT_DIR}. Please ensure the data files exist and are valid JSON.")
    
    return datasets.get("tool_memorization.json"), datasets.get("retrieval_training.json"), datasets.get("agent_tuning.json")

# Step 2: Prepare datasets
# Utility function to convert to Hugging Face Dataset format
def prepare_dataset(data):
    return Dataset.from_pandas(pd.DataFrame(data))

# Tokenization function without 'return_tensors'
def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples['description'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

# Step 3: Load base model and tokenizer
def load_base_model():
    print("Loading base model...")
    
    # QLoRA quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # QLoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Step 4: Define training functions
# Tool memorization fine-tuning
def fine_tune_tool_memorization(model, tokenizer, tool_memorization_data):
    print("Fine-tuning model for tool memorization...")
    dataset = Dataset.from_pandas(pd.DataFrame(tool_memorization_data))
    
    def tokenize_function(examples):
        return tokenizer(examples["description"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Use DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(FINETUNED_MODEL_DIR, "tool_memorization"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_steps=1000,
        logging_dir="logs",
        save_total_limit=1,
        remove_unused_columns=False,  # Important for causal language modeling
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"  # Use 8-bit AdamW
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(os.path.join(FINETUNED_MODEL_DIR, "tool_memorization"))
    tokenizer.save_pretrained(os.path.join(FINETUNED_MODEL_DIR, "tool_memorization"))
    
    # Clear memory
    del model, tokenizer, trainer, dataset, tokenized_dataset
    torch.cuda.empty_cache()
    gc.collect()
    return trainer

# Retrieval training fine-tuning
def fine_tune_retrieval_training(model, tokenizer, retrieval_training_data):
    print("Fine-tuning model for retrieval training...")
    dataset = prepare_dataset(retrieval_training_data)

    def tokenize_function(examples):
        # Tokenize both query and response
        queries = tokenizer(examples["query"], truncation=True, padding="max_length", max_length=128)
        responses = tokenizer(examples["response"], truncation=True, padding="max_length", max_length=128)
        
        # Combine query and response
        combined = {
            "input_ids": queries["input_ids"] + responses["input_ids"],
            "attention_mask": queries["attention_mask"] + responses["attention_mask"],
        }
        
        # Set labels to be the same as input_ids for causal language modeling
        combined["labels"] = combined["input_ids"].copy()
        
        return combined

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Use DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(FINETUNED_MODEL_DIR, "retrieval_training"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_steps=500,
        logging_dir="logs",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    # Define a custom trainer
    class CausalLMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(os.path.join(FINETUNED_MODEL_DIR, "retrieval_training"))

    # Clear memory
    del trainer, dataset, tokenized_dataset
    torch.cuda.empty_cache()
    gc.collect()
    return trainer

# End-to-end agent tuning fine-tuning
def fine_tune_agent_tuning(model, tokenizer, agent_tuning_data):
    print("Fine-tuning model for end-to-end agent tuning...")
    dataset = prepare_dataset(agent_tuning_data)

    def tokenize_function(examples):
        # Combine query and response into a single text
        texts = [f"Query: {q}\nResponse: {r}" for q, r in zip(examples["query"], examples["response"])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(FINETUNED_MODEL_DIR, "agent_tuning"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=16,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_steps=500,
        logging_dir="logs",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(os.path.join(FINETUNED_MODEL_DIR, "agent_tuning"))
    
    # Clear memory
    del trainer, dataset, tokenized_dataset
    torch.cuda.empty_cache()
    gc.collect()
    return trainer

def log_memory_usage(stage: str):
    print(f"Memory Usage at {stage}:")
    print(torch.cuda.memory_summary())

# Custom callback to log GPU memory usage
class GPUMemoryCallback(TrainerCallback):
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id

    def on_step_end(self, args, state, control, **kwargs):
        gpu = GPUtil.getGPUs()[self.gpu_id]
        memory_used = gpu.memoryUsed
        memory_total = gpu.memoryTotal
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        print(f"Step {state.global_step}: GPU Memory: {memory_used}/{memory_total} MB, CPU: {cpu_percent}%, RAM: {ram_percent}%")

    def on_evaluate(self, args, state, control, **kwargs):
        gpu = GPUtil.getGPUs()[self.gpu_id]
        memory_used = gpu.memoryUsed
        memory_total = gpu.memoryTotal
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        print(f"Evaluation: GPU Memory: {memory_used}/{memory_total} MB, CPU: {cpu_percent}%, RAM: {ram_percent}%")

gpu_memory_callback = GPUMemoryCallback()

def print_training_summary(trainer):
    print("\nTraining Summary:")
    print(f"Total training steps: {trainer.state.global_step}")
    print(f"Best evaluation loss: {trainer.state.best_metric}")
    print(f"Training time: {trainer.state.total_flos / 1e9:.2f} seconds")
    print(f"Average training speed: {trainer.state.global_step / (trainer.state.total_flos / 1e9):.2f} steps/second")

# Main execution
def main():
    try:
        tool_memorization_data, retrieval_training_data, agent_tuning_data = load_training_data()
        
        # Load the base model and tokenizer
        model, tokenizer = load_base_model()
        log_memory_usage("After Model Loading")

        if tool_memorization_data is not None:
            try:
                trainer = fine_tune_tool_memorization(model, tokenizer, tool_memorization_data)
                print_training_summary(trainer)
                log_memory_usage("After Tool Memorization Fine-Tuning")
            except Exception as e:
                print(f"Error during tool memorization fine-tuning: {e}")
                traceback.print_exc()
    
        if retrieval_training_data is not None:
            try:
                trainer = fine_tune_retrieval_training(model, tokenizer, retrieval_training_data)
                print_training_summary(trainer)
                log_memory_usage("After Retrieval Training Fine-Tuning")
            except Exception as e:
                print(f"Error during retrieval training fine-tuning: {e}")
                traceback.print_exc()
    
        if agent_tuning_data is not None:
            try:
                trainer = fine_tune_agent_tuning(model, tokenizer, agent_tuning_data)
                print_training_summary(trainer)
                log_memory_usage("After Agent Tuning Fine-Tuning")
            except Exception as e:
                print(f"Error during agent tuning fine-tuning: {e}")
                traceback.print_exc()
    
        print("All fine-tuning stages completed.")
    
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("Please check your data files and ensure they are in the correct format.")
        traceback.print_exc()

if __name__ == "__main__":
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    main()