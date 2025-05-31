"""
merge_lora.py

Merges LoRA (Low-Rank Adaptation) weights with a base model and saves the
resulting merged model and its tokenizer. This script is typically run after
fine-tuning a model with LoRA adapters.

Author: Justin Greisiger Frost <justinfrost@duck.com>
Date: 2025-05-31
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # Updated to Mistral
LORA_DIR = "models/mistral_7b_instruct_lora"  # Updated to match train.py's OUTPUT_DIR
MERGED_DIR = "models/mistral_7b_instruct_merged"  # Updated for merged Mistral model

logger.info("Loading base model...")


def main():
    """
    Main function to load the base model, apply LoRA adapters,
    merge them, and save the final model and tokenizer.
    """
    logger.info("--- Merge LoRA Script Started ---")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="auto"
    )
    logger.info("Base model loaded.")
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    logger.info("LoRA adapter loaded.")
    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()
    logger.info("LoRA weights merged and unloaded.")
    logger.info("Saving merged model...")
    model.save_pretrained(MERGED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    logger.info("Merged model and tokenizer saved to %s", MERGED_DIR)
    logger.info("--- Merge LoRA Script Finished ---")


if __name__ == "__main__":
    main()
