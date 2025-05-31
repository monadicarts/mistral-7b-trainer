from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # Updated to Mistral
LORA_DIR = "models/mistral_7b_instruct_lora"  # Updated to match train.py's OUTPUT_DIR
MERGED_DIR = "models/mistral_7b_instruct_merged"  # Updated for merged Mistral model

logger.info("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype="auto", device_map="auto"
)
logger.info("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_DIR)
logger.info("Merging LoRA weights...")
model = model.merge_and_unload()
logger.info("Saving merged model...")
model.save_pretrained(MERGED_DIR)
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
tokenizer.save_pretrained(MERGED_DIR)
logger.info("Done. Merged model saved to %s", MERGED_DIR)
