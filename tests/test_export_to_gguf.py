import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import torch  # For type hints

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import export_to_gguf  # Assuming export_to_gguf.py is in the parent directory


@pytest.fixture(autouse=True)
def mock_logger_export():
    with patch("export_to_gguf.logger") as mock_log_instance:
        yield mock_log_instance


@patch("export_to_gguf.FastLanguageModel")
@patch("export_to_gguf.shutil.copy")
@patch("export_to_gguf.os.path.exists")
@patch("export_to_gguf.os.path.dirname")
@patch("export_to_gguf.requests.get")  # Mock requests for config download
@patch("export_to_gguf.gc.collect")
@patch("torch.cuda.empty_cache")
@patch(
    "torch.cuda.is_available", return_value=True
)  # Assume CUDA is available for empty_cache call
def test_export_gguf_main_flow_config_exists(
    mock_cuda_is_available,
    mock_empty_cache,
    mock_gc_collect,
    mock_requests_get,
    mock_os_path_dirname,
    mock_os_path_exists,
    mock_shutil_copy,
    mock_fast_lang_model,
    mock_logger_export,
):
    # --- Mock Unsloth ---
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_fast_lang_model.from_pretrained.return_value = (
        mock_model_instance,
        mock_tokenizer_instance,
    )

    # --- Mock file system for config.json ---
    # Scenario 1: output_config_path does NOT exist, base_config_path DOES exist
    def side_effect_exists(path):
        if path == os.path.join(
            export_to_gguf.GGUF_OUTPUT, "config.json"
        ):  # Assuming GGUF_OUTPUT is a file path
            return False
        if path == f"{export_to_gguf.OUTPUT_DIR}/config.json":
            return True
        return False  # Default for other paths if any

    mock_os_path_exists.side_effect = side_effect_exists
    mock_os_path_dirname.return_value = os.path.dirname(export_to_gguf.GGUF_OUTPUT)

    # --- Call the main function ---
    export_to_gguf.main()

    # --- Assertions ---
    # Model loading
    mock_fast_lang_model.from_pretrained.assert_called_once_with(
        model_name=export_to_gguf.BASE_MODEL,
        max_seq_length=export_to_gguf.MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=False,
        device_map="auto",
    )

    # GGUF saving
    mock_model_instance.save_pretrained_gguf.assert_called_once_with(
        export_to_gguf.GGUF_OUTPUT,
        mock_tokenizer_instance,
        quantization_method=export_to_gguf.GGUF_QUANT,
    )

    # Config file handling
    gguf_dir = os.path.dirname(export_to_gguf.GGUF_OUTPUT)
    output_config_path = os.path.join(gguf_dir, "config.json")
    base_config_path = f"{export_to_gguf.OUTPUT_DIR}/config.json"

    mock_os_path_exists.assert_any_call(output_config_path)
    mock_os_path_exists.assert_any_call(base_config_path)
    mock_shutil_copy.assert_called_once_with(base_config_path, output_config_path)
    mock_requests_get.assert_not_called()  # Should not download if local copy works

    # Memory cleanup
    mock_gc_collect.assert_called_once()
    mock_empty_cache.assert_called_once()
    mock_logger_export.info.assert_any_call("Memory cleanup complete.")


@patch("export_to_gguf.FastLanguageModel")
@patch("export_to_gguf.shutil.copy")
@patch("export_to_gguf.os.path.exists")
@patch("export_to_gguf.os.path.dirname")
@patch("export_to_gguf.requests.get")
@patch(
    "builtins.open", new_callable=mock_open
)  # Mock open for writing downloaded config
@patch("export_to_gguf.gc.collect")
@patch("torch.cuda.empty_cache")
@patch("torch.cuda.is_available", return_value=True)
def test_export_gguf_main_flow_config_download(
    mock_cuda_is_available,
    mock_empty_cache,
    mock_gc_collect,
    mock_builtin_open,
    mock_requests_get,
    mock_os_path_dirname,
    mock_os_path_exists,
    mock_shutil_copy,
    mock_fast_lang_model,
    mock_logger_export,
):
    mock_fast_lang_model.from_pretrained.return_value = (MagicMock(), MagicMock())

    # Scenario 2: output_config_path and base_config_path do NOT exist -> download
    mock_os_path_exists.return_value = False  # All paths don't exist initially
    mock_os_path_dirname.return_value = os.path.dirname(export_to_gguf.GGUF_OUTPUT)

    # Mock requests.get
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.content = b'{"config_key": "config_value"}'
    mock_requests_get.return_value = mock_response

    export_to_gguf.main()

    gguf_dir = os.path.dirname(export_to_gguf.GGUF_OUTPUT)
    output_config_path = os.path.join(gguf_dir, "config.json")

    mock_shutil_copy.assert_not_called()
    mock_requests_get.assert_called_once()  # Check the URL if it's static, or use mock.ANY
    # Example: mock_requests_get.assert_called_once_with("https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/config.json")
    mock_builtin_open.assert_called_once_with(output_config_path, "wb")
    mock_builtin_open().write.assert_called_once_with(b'{"config_key": "config_value"}')
    mock_logger_export.info.assert_any_call(f"Downloaded config.json to {gguf_dir}")
