from unsloth import FastLanguageModel

import gc
import os
import shutil

import torch

from utils.logger import get_logger

logger = get_logger(__name__)

BASE_MODEL = (
    "models/mistral_7b_instruct_merged"  # Use the new merged Mistral model directory
)
OUTPUT_DIR = "models/mistral_7b_instruct_merged"  # Directory containing the merged model's config
GGUF_OUTPUT = "models/mistral_7b_instruct_ollama.gguf"  # Updated GGUF output filename
GGUF_QUANT = "q4_k_m"
MAX_SEQ_LENGTH = (
    2048  # Aligned with training, or consider 4096 for Mistral's capabilities
)


def main():
    try:
        logger.info("Loading base model and tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.float16,
            load_in_4bit=False,
            device_map="auto",
        )
        logger.info("Model and tokenizer loaded.")

        logger.info(
            f"Exporting to GGUF: {GGUF_OUTPUT} (quantization: {GGUF_QUANT}) ..."
        )
        model.save_pretrained_gguf(
            GGUF_OUTPUT,
            tokenizer,
            quantization_method=GGUF_QUANT,
        )
        logger.info(f"Model successfully exported to {GGUF_OUTPUT}")

        # --- Ensure config.json is present in GGUF_OUTPUT directory ---
        gguf_config_dir = (
            GGUF_OUTPUT if os.path.isdir(GGUF_OUTPUT) else os.path.dirname(GGUF_OUTPUT)
        )
        output_config_path = os.path.join(gguf_config_dir, "config.json")
        base_config_path = f"{OUTPUT_DIR}/config.json"
        try:
            if not os.path.exists(output_config_path):
                if os.path.exists(base_config_path):
                    shutil.copy(base_config_path, output_config_path)
                    logger.info(f"Copied config.json to {gguf_config_dir}")
                else:
                    # Try to download from Hugging Face if not present locally
                    import requests

                    url = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/config.json"
                    r = requests.get(url)
                    r.raise_for_status()
                    with open(output_config_path, "wb") as f:
                        f.write(r.content)
                    logger.info(f"Downloaded config.json to {gguf_config_dir}")
        except Exception as e:
            logger.error(f"Could not ensure config.json in {gguf_config_dir}: {e}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
    finally:
        logger.info("Cleaning up memory...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Memory cleanup complete.")


if __name__ == "__main__":
    main()
