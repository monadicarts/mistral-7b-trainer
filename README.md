# Mistral 7b v0.3 LLM Model Trainer

This project provides a comprehensive pipeline for fine-tuning the **Mistral 7B Instruct v0.3** Large Language Model (**LLM**) using Low-Rank Adaptation (**LoRA**), merging the trained **LoRA** adapters with the base model, and subsequently exporting the merged model to the **GGUF** format for efficient inference with tools like **llama.cpp** or **Ollama**.

**Author:** [Justin Greisiger Frost](https://github.com/gneissguise)

## Table of Contents

- [Mistral 7b v0.3 LLM Model Trainer](#mistral-7b-v03-llm-model-trainer)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create and Activate Virtual Environment](#2-create-and-activate-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Step 1: Prepare Your Training Data](#step-1-prepare-your-training-data)
    - [Step 2: Fine-tune the Model (LoRA Training)](#step-2-fine-tune-the-model-lora-training)
    - [Step 3: Merge LoRA Adapters](#step-3-merge-lora-adapters)
    - [Step 4: Export to GGUF Format](#step-4-export-to-gguf-format)
  - [Testing](#testing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Overview

The goal of this project is to streamline the process of customizing a powerful base LLM like [**Mistral 7B Instruct v0.3**](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for specific tasks. It leverages the Unsloth library for efficient **LoRA** fine-tuning and provides scripts to manage the model lifecycle from training to deployment-ready **GGUF** format.

The pipeline consists of three main stages:
1.  **Training (`train.py`):** Fine-tunes the base model on a custom dataset using LoRA, saving the adapter weights.
2.  **Merging (`merge_lora.py`):** Combines the trained LoRA adapters with the original base model to create a new, full-weight fine-tuned model.
3.  **Exporting (`export_to_gguf.py`):** Converts the merged model into the GGUF format, including quantization, for use in various inference engines.

## Features

- Fine-tuning with **Mistral 7B Instruct v0.3** as the base model.
- Efficient **LoRA** training using **Unsloth**.
- Customizable training parameters (batch size, learning rate, epochs, **LoRA** R/Alpha, etc.).
- Script to merge **LoRA** adapters with the base model.
- Script to export the merged model to **GGUF** format with specified quantization.
- Automated handling of `config.json` for **GGUF** export.
- Comprehensive logging throughout the process.
- Unit tests for core functionalities.

## Project Structure

```bash
.
├── data/
├── models/                       # Output directory for trained models
├── tests/                        # Pytest unit tests
│   ├── test_train.py
│   ├── test_merge_lora.py
│   └── test_export_to_gguf.py
├── utils/
│   └── logger.py                 # Logging utility
├── .gitignore
├── export_to_gguf.py             # Script to export model to GGUF
├── merge_lora.py                 # Script to merge LoRA adapters
├── pytest.ini                    # Pytest configuration
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── train.py                      # Main training script
```

## Prerequisites

- Python 3.9+ (Python 3.11 recommended, as used in development)
- NVIDIA GPU with CUDA support (for efficient training with Unsloth and bitsandbytes)
- Sufficient VRAM (Mistral 7B fine-tuning can be demanding, typically requires at least 16GB VRAM. Tested on a 24GB VRAM Ampere GPU)
- `git` for cloning the repository.

## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:monadicarts/mistral-7b-trainer.git
cd mistral-7b-trainer
```

### 2. Create and Activate Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# Using venv (replace .venv with your preferred name like 'bin')
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Configuration

Each script (`train.py`, `merge_lora.py`, `export_to_gguf.py`) has configuration constants defined at the top of the file. You can modify these to suit your needs:

**`train.py`:**
- `BASE_MODEL`: The Hugging Face model ID for the base model.
- `TRAIN_DATA_PATH`: Path to your training data file (JSONL format expected).
- `OUTPUT_DIR`: Directory to save the trained LoRA adapters.
- `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `GRAD_ACCUM`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LR_SCHEDULER_TYPE`, `EARLY_STOPPING_PATIENCE`: Training and LoRA hyperparameters.

**`merge_lora.py`:**
- `BASE_MODEL`: Should match the base model used in `train.py`.
- `LORA_DIR`: Path to the saved LoRA adapters (output from `train.py`).
- `MERGED_DIR`: Directory to save the merged model.

**`export_to_gguf.py`:**
- `BASE_MODEL`: Path to the merged model directory (output from `merge_lora.py`).
- `OUTPUT_DIR`: Directory containing the merged model's `config.json` (usually same as `BASE_MODEL` for merged models).
- `GGUF_OUTPUT`: Full path for the output GGUF file.
- `GGUF_QUANT`: Quantization method (e.g., "q4_k_m").
- `MAX_SEQ_LENGTH`: Max sequence length for the model.

## Usage

Follow these steps to fine-tune, merge, and export your model.

### Step 1: Prepare Your Training Data

Your training data should be in a JSON Lines (`.jsonl`) format, where each line is a JSON object. The `train.py` script expects each JSON object to have an `instruction` field and an `output` field for supervised fine-tuning.

Example line in `data/train_data_cleaned.jsonl`:
```json
{"instruction": "Translate the following English text to French: 'Hello, world!'", "output": "Bonjour, le monde !"}
```
Place your prepared data file at the path specified by `TRAIN_DATA_PATH` in `train.py`.

### Step 2: Fine-tune the Model (LoRA Training)

Run the `train.py` script to start the fine-tuning process.

```bash
python train.py
```

This will:
- Load the base model and tokenizer.
- Configure LoRA adapters.
- Load and format your dataset.
- Train the model using the specified hyperparameters.
- Save the trained LoRA adapters to the directory specified by `OUTPUT_DIR` (e.g., `models/mistral_7b_instruct_lora/`).

Monitor the console output for training progress and logs.

### Step 3: Merge LoRA Adapters

After training, merge the LoRA adapters with the base model.

```bash
python merge_lora.py
```

This script will:
- Load the base model.
- Load the LoRA adapters from `LORA_DIR`.
- Merge the adapters into the base model.
- Save the full merged model and its tokenizer to `MERGED_DIR` (e.g., `models/mistral_7b_instruct_merged/`).

### Step 4: Export to GGUF Format

Finally, convert the merged model to GGUF format.

```bash
python export_to_gguf.py
```

This will:
- Load the merged model.
- Convert it to GGUF format using the specified quantization method (`GGUF_QUANT`).
- Save the GGUF file to the path specified by `GGUF_OUTPUT` (e.g., `models/mistral_7b_instruct_ollama.gguf`).
- Ensure `config.json` is present in the GGUF output directory, copying it from the merged model directory or downloading it if necessary.

## Testing

The project includes unit tests for the core scripts. To run the tests, ensure `pytest` is installed (it's in `requirements.txt`) and run:

```bash
python -m pytest
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- 🦥 The [Unsloth](https://unsloth.ai/) team for their efficient fine-tuning library.
- 🤗 The [Hugging Face](https://huggingface.co/) team for `transformers`, `datasets`, and `peft`.
- 🦙 The [llama.cpp](https://github.com/ggerganov/llama.cpp) team for providing community for the GGUF format and tools.
- ✨ The [Google Gemini](https://gemini.google.com) team for helping me get started with this project and providing valuable insights.
- 🌬️ The [Mistral](https://www.mistral.ai/) team for providing the Mistral 7B Instruct v0.3 model
- ❤️ The [open-source community](https://opensource.org/) for their contributions and support!
