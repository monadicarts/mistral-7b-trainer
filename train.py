"""
train.py

Fine-tuning script for LLMs using Unsloth and TRL SFTTrainer.
Handles dataset loading, formatting, LoRA configuration, training, and saving.

Author: Justin Greisiger Frost <justinfrost@duck.com>
Date: 2025-05-31
"""

import unsloth  # Import unsloth first

import gc
import os
import random
import time

import torch
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)

# --- Configuration ---
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # Changed to Mistral Instruct
TRAIN_DATA_PATH = "data/train_data_cleaned.jsonl"
OUTPUT_DIR = "models/mistral_7b_instruct_lora"  # Updated for Mistral
MAX_SEQ_LENGTH = 2048  # Increased for better context handling
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05  # Added LoRA dropout for regularization
LR_SCHEDULER_TYPE = "cosine"
EARLY_STOPPING_PATIENCE = 2

logger.info("--- Training Configuration ---")
logger.info(f"BASE_MODEL: {BASE_MODEL}")
logger.info(f"TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
logger.info(f"BATCH_SIZE (per device): {BATCH_SIZE}")
logger.info(f"GRADIENT_ACCUMULATION_STEPS: {GRAD_ACCUM}")
logger.info(f"EFFECTIVE_BATCH_SIZE: {BATCH_SIZE * GRAD_ACCUM}")
logger.info(f"EPOCHS: {EPOCHS}")
logger.info(f"LEARNING_RATE: {LEARNING_RATE}")
logger.info(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
logger.info(f"LORA_R: {LORA_R}")
logger.info(f"LORA_ALPHA: {LORA_ALPHA}")
logger.info(f"LORA_DROPOUT: {LORA_DROPOUT}")
logger.info(f"LR_SCHEDULER_TYPE: {LR_SCHEDULER_TYPE}")
logger.info(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
logger.info(f"FP16 enabled: {not torch.cuda.is_bf16_supported()}")
logger.info(f"BF16 enabled: {torch.cuda.is_bf16_supported()}")
logger.info("--- End Training Configuration ---")


def formatting_prompts_func(example):
    """
    Formats a dataset example into the prompt format for supervised fine-tuning.

    Args:
        example (dict): A dictionary potentially containing 'instruction' and 'output' keys.

    Returns:
        dict: A dictionary with a single 'text' key containing the formatted prompt.
    """
    instruction = example.get("instruction", "")
    # Ensure output is a string; if it's None or not a string, treat as empty.
    output_val = example.get("output")
    output = str(output_val) if output_val is not None else ""
    prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
    return {"text": prompt}


def main():
    """
    Main function to orchestrate the fine-tuning process.
    This includes setting the random seed, loading the base model and tokenizer,
    configuring LoRA, preparing the dataset, setting up training arguments,
    running the SFTTrainer, and saving the trained LoRA adapters.
    """
    logger.info("--- Training Script Started ---")

    # --- Dynamic random seed ---
    # Moved into main() for better testability and control
    global SEED  # Declare SEED as global if it's used elsewhere in the module, or pass it around.
    # For this script structure, making it accessible for logging/LoRA config is fine.
    SEED = int(
        os.environ.get("TRAIN_SEED", str(int(time.time()) + random.randint(0, 100000)))
    )
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logger.info(f"SEED used for this run: {SEED}")

    # --- Load Model ---
    logger.info("Preparing to load base model and tokenizer...")
    _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _load_in_4bit = True
    logger.info(
        f"Attempting to load: model_name='{BASE_MODEL}', max_seq_length={MAX_SEQ_LENGTH}, dtype={_dtype}, load_in_4bit={_load_in_4bit}"
    )
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=_dtype,
        load_in_4bit=_load_in_4bit,
    )
    logger.info("Model and tokenizer loaded successfully.")

    # --- Prepare LoRA ---
    logger.info("Configuring model for LoRA fine-tuning...")
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
    )
    logger.info("LoRA configuration complete.")

    # --- Load and Format Dataset ---
    logger.info(f"Loading dataset from {TRAIN_DATA_PATH}...")
    raw_dataset = load_dataset(
        "json", data_files={"train": TRAIN_DATA_PATH}, split="train"
    )
    logger.info(f"Loaded {len(raw_dataset)} training samples.")

    # Filter out empty or very short outputs
    raw_dataset = raw_dataset.filter(
        lambda ex: ex.get("output", None)
        and isinstance(ex.get("output"), str)
        and len(ex["output"].strip()) > 10
    )
    logger.info(
        f"Filtered dataset: {len(raw_dataset)} samples remain after removing empty/short outputs."
    )

    # Shuffle and split into train/validation
    raw_dataset = raw_dataset.shuffle(seed=SEED)
    split = raw_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(
        f"Dataset split: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples."
    )

    logger.info("Formatting dataset for supervised fine-tuning...")
    train_dataset = train_dataset.map(
        formatting_prompts_func, desc="Formatting train prompts"
    )
    eval_dataset = eval_dataset.map(
        formatting_prompts_func, desc="Formatting eval prompts"
    )

    logger.info("Showing a few examples of formatted training data (first 3):")
    # Calculate total training steps for potential warmup_steps adjustment
    effective_batch_size = BATCH_SIZE * GRAD_ACCUM
    total_training_steps = 0
    if len(train_dataset) > 0 and effective_batch_size > 0 and EPOCHS > 0:
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_training_steps = steps_per_epoch * EPOCHS
        suggested_warmup_steps = int(0.1 * total_training_steps)  # 10% of total steps
        logger.info(
            f"Calculated total training steps: {total_training_steps}. Suggested warmup_steps (10%): {suggested_warmup_steps}"
        )
    for i in range(min(3, len(train_dataset))):
        logger.info(f"Example {i + 1}: {train_dataset[i]['text'][:250]}...")

    logger.info("Dataset formatting complete.")

    # --- Training ---
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=SEED,
        eval_strategy="epoch",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        warmup_steps=int(total_training_steps * 0.05)
        if total_training_steps > 0
        else 0,  # Ensure warmup_steps is an integer
    )

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # Tell SFTTrainer to use the 'text' field
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    logger.info("Setting os.environ['UNSLOTH_RETURN_LOGITS'] = '1' for Unsloth.")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # --- Save LoRA adapters and tokenizer ---
    logger.info(f"Saving LoRA adapters and tokenizer to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("LoRA adapters and tokenizer saved.")

    # --- Free memory ---
    logger.info("Freeing memory...")
    del trainer
    del model
    del tokenizer
    del train_dataset
    del eval_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory freed. Training script complete.")


if __name__ == "__main__":
    main()
