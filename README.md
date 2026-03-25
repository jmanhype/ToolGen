# ToolGen

Fine-tunes Llama-3.2-1B for tool usage using the ToolBench dataset. 3 training stages, LoRA/QLoRA, single script.

## What It Does

Takes the ToolBench dataset and fine-tunes a language model to:

1. **Memorize tools** -- learn tool descriptions and capabilities
2. **Retrieve tools** -- match user queries to relevant tools
3. **Act as agent** -- use tools end-to-end in conversation

## Technical Details

| Parameter | Value |
|-----------|-------|
| Base model | unsloth/Llama-3.2-1B |
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| Quantization | 4-bit (QLoRA) |
| Dataset | ToolBench (3 splits) |
| Training script | `scripts/train_toolgen.py` |

## Project Structure

```
scripts/
  train_toolgen.py              # Main training pipeline (all 3 stages)
  setup_toolbench_data.py       # Dataset preparation
  processed_toolbench/          # Preprocessed JSON files
    tool_memorization.json
    retrieval_training.json
    agent_tuning.json
finetuned_toolgen_model/
  tool_memorization/            # LoRA adapter weights + tokenizer
src/
  toolgen/                      # Package source
  utils/                        # File and token utilities
examples/
  agent_tuning_example.py
  tool_retrieval_example.py
tests/
  test_data_preprocessing.py
  test_tool_memorization.py
```

## Requirements

- Python 3.8+
- CUDA GPU
- GCC/G++ 10

## Setup

```bash
git clone https://github.com/jmanhype/ToolGen.git
cd ToolGen
pip install -r requirements.txt
```

## Usage

```bash
# Prepare data
python scripts/setup_toolbench_data.py

# Run all 3 training stages
python scripts/train_toolgen.py
```

## Status

The repository includes a trained LoRA adapter checkpoint (checkpoint-3894) for the tool memorization stage. Retrieval training and agent tuning stages exist in the training script but their completion status is unclear. No evaluation metrics are published.

## License

See LICENSE file.
