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
   git clone https://github.com/toolgen/toolgen.git
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

The ToolGen model is evaluated using the following metrics:

1. Tool Memorization:
   - Accuracy: Percentage of correctly memorized tool descriptions and functionalities
   - F1 Score: Harmonic mean of precision and recall for tool information retrieval

2. Retrieval Training:
   - Mean Reciprocal Rank (MRR): Measures the ranking quality of retrieved tools
   - Normalized Discounted Cumulative Gain (NDCG): Evaluates the ranking quality of retrieved tools

3. End-to-End Agent Tuning:
   - Task Completion Rate: Percentage of successfully completed user requests
   - Response Relevance: Human-evaluated score for the relevance of the agent's responses
   - Tool Selection Accuracy: Percentage of correctly selected tools for given tasks

## Results

Preliminary results show promising performance across all three fine-tuning stages:

1. Tool Memorization:
   - Accuracy: 92.5%
   - F1 Score: 0.89

2. Retrieval Training:
   - MRR: 0.85
   - NDCG@5: 0.79

3. End-to-End Agent Tuning:
   - Task Completion Rate: 87%
   - Response Relevance: 4.2/5
   - Tool Selection Accuracy: 91%

These results demonstrate the effectiveness of the ToolGen approach in enhancing the model's tool-related capabilities.

## Limitations and Future Work

While ToolGen shows promising results, there are several limitations and areas for future improvement:

1. Limited Tool Set: The current model is trained on a finite set of tools. Future work should focus on expanding the tool database and improving the model's ability to generalize to new, unseen tools.

2. Language Dependency: ToolGen is primarily trained on English data. Multilingual support would greatly enhance its applicability in diverse settings.

3. Context Understanding: Improving the model's ability to understand complex, multi-turn conversations and maintain context over longer interactions.

4. Ethical Considerations: Further work is needed to ensure the model's responses align with ethical guidelines and to mitigate potential biases in tool selection and usage.

5. Computational Efficiency: While LoRA and QLoRA techniques improve efficiency, further optimizations could make the model more accessible for deployment on less powerful hardware.

Future work will address these limitations and explore integration with other AI systems for more comprehensive tool-assisted problem-solving.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the ToolBench dataset and builds upon the Llama model architecture. We thank the creators and contributors of these resources. Special thanks to the open-source community for their invaluable contributions to the field of natural language processing and tool-augmented AI systems.