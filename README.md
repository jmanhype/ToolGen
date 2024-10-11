# ToolGen

ToolGen is a project for fine-tuning language models on tool-related tasks using the ToolBench dataset. It aims to enhance the model's ability to understand, retrieve, and use various tools in a conversational AI context.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Setup](#setup)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Running the Fine-tuning Process](#running-the-fine-tuning-process)
6. [Fine-tuning Stages](#fine-tuning-stages)
7. [Output](#output)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Limitations and Future Work](#limitations-and-future-work)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

## Project Overview

ToolGen fine-tunes a base language model (Llama-3.2-1B) for three specific tasks:
1. Tool Memorization: Enhancing the model's knowledge of various tools and their functionalities.
2. Retrieval Training: Improving the model's ability to retrieve relevant tools based on user queries.
3. End-to-End Agent Tuning: Fine-tuning the model to act as an agent that can understand user requests and use appropriate tools to fulfill them.

## Model Architecture

ToolGen uses the Llama-3.2-1B model as its base and applies the LoRA (Low-Rank Adaptation) technique for efficient fine-tuning. This approach allows for faster training and reduced memory requirements while maintaining performance.

Key features of the model architecture:
- Base Model: unsloth/Llama-3.2-1B
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit quantization using QLoRA

## Dataset

ToolGen uses the ToolBench dataset, which contains:
- Tool descriptions and functionalities for tool memorization
- User queries and relevant tool mappings for retrieval training
- Conversational data for end-to-end agent training

The dataset is preprocessed and split into three JSON files corresponding to each training stage.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- GCC 10 and G++ 10

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/toolgen.git
   cd toolgen
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the correct versions of GCC and G++:
   ```
   sudo apt-get install gcc-10 g++-10
   ```
   Or using Conda:
   ```
   conda install -c conda-forge gcc_linux-64=10 gxx_linux-64=10
   ```

## Usage

### Data Preparation

Before running the fine-tuning process, ensure that your ToolBench dataset is properly formatted and located in the `scripts/processed_toolbench` directory. The script expects the following files:

- `tool_memorization.json`
- `retrieval_training.json`
- `agent_tuning.json`

### Running the Fine-tuning Process

To run the complete fine-tuning process:
```
python scripts/train_toolgen.py
```

This script will sequentially perform tool memorization, retrieval training, and end-to-end agent tuning.

## Fine-tuning Stages

1. Tool Memorization: Trains the model to understand and remember tool descriptions and functionalities.
2. Retrieval Training: Improves the model's ability to retrieve relevant tools based on user queries.
3. End-to-End Agent Tuning: Fine-tunes the model to act as an agent, understanding user requests and using appropriate tools.

## Output

The fine-tuned models will be saved in the `finetuned_toolgen_model` directory, with separate subdirectories for each stage:

- `finetuned_toolgen_model/tool_memorization`
- `finetuned_toolgen_model/retrieval_training`
- `finetuned_toolgen_model/agent_tuning`

## Evaluation

[Add information about how the model is evaluated, including metrics used]

## Results

[Include any available results or performance metrics]

## Limitations and Future Work

[Discuss any known limitations of the current approach and potential areas for improvement]

## License

[Specify the license for the project]

## Acknowledgements

This project uses the ToolBench dataset and builds upon the Llama model architecture. We thank the creators and contributors of these resources.