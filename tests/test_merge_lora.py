import pytest
from unittest.mock import patch, MagicMock
import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import merge_lora  # Assuming merge_lora.py is in the parent directory


@pytest.fixture(autouse=True)
def mock_logger_merge():
    with patch("merge_lora.logger") as mock_log_instance:
        yield mock_log_instance


@patch("merge_lora.AutoModelForCausalLM")
@patch("merge_lora.PeftModel")
@patch("merge_lora.AutoTokenizer")
def test_merge_lora_main_flow(
    mock_auto_tokenizer, mock_peft_model, mock_auto_model, mock_logger_merge
):
    # --- Mock Hugging Face and PEFT models/tokenizers ---
    mock_base_model_instance = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_base_model_instance

    mock_peft_adapter_instance = MagicMock()
    # This mock will be the one that has merge_and_unload and save_pretrained
    mock_merged_model_instance = MagicMock()
    mock_peft_adapter_instance.merge_and_unload.return_value = (
        mock_merged_model_instance
    )
    mock_peft_model.from_pretrained.return_value = mock_peft_adapter_instance

    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    # --- Call the main function ---
    merge_lora.main()

    # --- Assertions ---
    # Base model loading
    mock_auto_model.from_pretrained.assert_called_once_with(
        merge_lora.BASE_MODEL, torch_dtype="auto", device_map="auto"
    )

    # LoRA adapter loading
    mock_peft_model.from_pretrained.assert_called_once_with(
        mock_base_model_instance, merge_lora.LORA_DIR
    )

    # Merging
    mock_peft_adapter_instance.merge_and_unload.assert_called_once()

    # Saving merged model and tokenizer
    mock_merged_model_instance.save_pretrained.assert_called_once_with(
        merge_lora.MERGED_DIR
    )
    mock_auto_tokenizer.from_pretrained.assert_called_once_with(merge_lora.LORA_DIR)
    mock_tokenizer_instance.save_pretrained.assert_called_once_with(
        merge_lora.MERGED_DIR
    )

    # Check logging
    mock_logger_merge.info.assert_any_call("--- Merge LoRA Script Started ---")
    mock_logger_merge.info.assert_any_call(
        "Merged model and tokenizer saved to %s", merge_lora.MERGED_DIR
    )
    mock_logger_merge.info.assert_any_call("--- Merge LoRA Script Finished ---")
