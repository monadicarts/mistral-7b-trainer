import pytest
from unittest.mock import patch, MagicMock, call
import os
import torch  # Import torch for type hints and constants if needed by the script

# Import the functions/main from the script to be tested
# We assume train.py is in the parent directory or accessible via PYTHONPATH
# For simplicity, if train.py is in the same directory as the tests directory's parent:
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import train


# Mock the logger at the module level where it's used in train.py
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("train.logger") as mock_log_instance:
        yield mock_log_instance


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    # Set a fixed seed for reproducibility in tests
    monkeypatch.setenv("TRAIN_SEED", "42")
    # Mock os.environ.get for other potential uses if necessary
    original_os_environ_get = os.environ.get

    def mock_os_get(key, default=None):
        if key == "TRAIN_SEED":
            return "42"
        return original_os_environ_get(key, default)

    monkeypatch.setattr(os.environ, "get", mock_os_get)


def test_formatting_prompts_func():
    example1 = {"instruction": "Translate to French", "output": "Hello"}
    assert train.formatting_prompts_func(example1) == {
        "text": "<s>[INST] Translate to French [/INST] Hello</s>"
    }

    example2 = {"instruction": "Summarize", "output": "This is a long text."}
    assert train.formatting_prompts_func(example2) == {
        "text": "<s>[INST] Summarize [/INST] This is a long text.</s>"
    }

    example_no_instruction = {"output": "Just an output."}
    assert train.formatting_prompts_func(example_no_instruction) == {
        "text": "<s>[INST]  [/INST] Just an output.</s>"
    }

    example_no_output = {"instruction": "Question?"}
    assert train.formatting_prompts_func(example_no_output) == {
        "text": "<s>[INST] Question? [/INST] </s>"
    }

    example_empty = {}
    assert train.formatting_prompts_func(example_empty) == {
        "text": "<s>[INST]  [/INST] </s>"
    }

    example_output_none = {"instruction": "Handle None", "output": None}
    assert train.formatting_prompts_func(example_output_none) == {
        "text": "<s>[INST] Handle None [/INST] </s>"
    }


@patch("train.unsloth.FastLanguageModel")
@patch("train.load_dataset")
@patch("train.TrainingArguments")
@patch("train.SFTTrainer")
@patch("train.EarlyStoppingCallback")
@patch("torch.manual_seed")
@patch("torch.cuda.manual_seed_all")
@patch(
    "torch.cuda.is_available", return_value=True
)  # Control CUDA availability for the test
@patch("torch.cuda.is_bf16_supported", return_value=True)  # Assume bf16 is supported
@patch("torch.cuda.empty_cache")
@patch("train.gc.collect")
def test_train_main_flow(
    # Mocks are injected in reverse order of @patch decorators (bottom-up)
    mock_train_gc_collect,  # Patches "train.gc.collect"
    mock_torch_cuda_empty_cache,  # Patches "torch.cuda.empty_cache"
    mock_torch_cuda_is_bf16_supported,  # Patches "torch.cuda.is_bf16_supported"
    mock_torch_cuda_is_available,  # Patches "torch.cuda.is_available"
    mock_torch_cuda_manual_seed_all,  # Patches "torch.cuda.manual_seed_all"
    mock_torch_manual_seed,  # Patches "torch.manual_seed"
    mock_early_stopping_callback,  # Patches "train.EarlyStoppingCallback"
    mock_sft_trainer,  # Patches "train.SFTTrainer"
    mock_training_args,  # Patches "train.TrainingArguments"
    mock_load_dataset,  # Patches "train.load_dataset"
    mock_fast_lang_model,  # Patches "train.unsloth.FastLanguageModel"
    mock_logger,  # from fixture
):
    # --- Mock Unsloth ---
    mock_model_instance = MagicMock()
    mock_tokenizer_instance = MagicMock()
    mock_peft_model_instance = MagicMock()

    # from_pretrained returns a tuple (model, tokenizer)
    mock_fast_lang_model.from_pretrained.return_value = (
        mock_model_instance,
        mock_tokenizer_instance,
    )
    mock_fast_lang_model.get_peft_model.return_value = mock_peft_model_instance

    # --- Mock Datasets ---
    mock_raw_dataset = MagicMock()
    mock_raw_dataset.__len__.return_value = 100  # Example length
    # Mock the filter method to return a new mock dataset (or itself if chaining)
    mock_filtered_dataset = MagicMock()
    mock_filtered_dataset.__len__.return_value = 90  # Length after filter
    mock_raw_dataset.filter.return_value = mock_filtered_dataset

    # Mock shuffle and train_test_split
    mock_shuffled_dataset = MagicMock()
    mock_filtered_dataset.shuffle.return_value = mock_shuffled_dataset

    mock_train_ds = MagicMock()
    mock_train_ds.__len__.return_value = 81  # 90 * 0.9
    mock_train_ds.map.return_value = mock_train_ds  # map returns itself for chaining
    mock_train_ds.__getitem__.return_value = {"text": "sample text"}

    mock_eval_ds = MagicMock()
    mock_eval_ds.__len__.return_value = 9  # 90 * 0.1
    mock_eval_ds.map.return_value = mock_eval_ds

    mock_shuffled_dataset.train_test_split.return_value = {
        "train": mock_train_ds,
        "test": mock_eval_ds,
    }
    mock_load_dataset.return_value = mock_raw_dataset

    # --- Mock Trainer ---
    mock_trainer_instance = MagicMock()
    mock_sft_trainer.return_value = mock_trainer_instance
    mock_early_stopping_cb_instance = MagicMock()
    mock_early_stopping_callback.return_value = mock_early_stopping_cb_instance

    # --- Call the main function ---
    train.main()

    # --- Assertions ---
    # Seed setting
    expected_seed = 42  # From monkeypatch

    # Assert torch.manual_seed(expected_seed) was called
    mock_torch_manual_seed.assert_called_with(expected_seed)

    # Assert torch.cuda.is_available() was called (it's patched to return True)
    mock_torch_cuda_is_available.assert_any_call()  # Called for seeding, and later for empty_cache

    # Assert torch.cuda.manual_seed_all(expected_seed) was called
    mock_torch_cuda_manual_seed_all.assert_called_with(expected_seed)

    # Model loading
    mock_fast_lang_model.from_pretrained.assert_called_once_with(
        model_name=train.BASE_MODEL,
        max_seq_length=train.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16
        if mock_torch_cuda_is_bf16_supported.return_value
        else torch.float16,
        load_in_4bit=True,
    )
    mock_fast_lang_model.get_peft_model.assert_called_once_with(
        mock_model_instance,  # The model instance from from_pretrained
        r=train.LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=train.LORA_ALPHA,
        lora_dropout=train.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=expected_seed,
    )

    # Dataset loading and processing
    mock_load_dataset.assert_called_once_with(
        "json", data_files={"train": train.TRAIN_DATA_PATH}, split="train"
    )
    mock_raw_dataset.filter.assert_called_once()  # Check that filter was called
    mock_filtered_dataset.shuffle.assert_called_once_with(seed=expected_seed)
    mock_shuffled_dataset.train_test_split.assert_called_once_with(
        test_size=0.1, seed=expected_seed
    )

    mock_train_ds.map.assert_called_once_with(
        train.formatting_prompts_func, desc="Formatting train prompts"
    )
    mock_eval_ds.map.assert_called_once_with(
        train.formatting_prompts_func, desc="Formatting eval prompts"
    )

    # TrainingArguments
    effective_batch_size = train.BATCH_SIZE * train.GRAD_ACCUM
    steps_per_epoch = mock_train_ds.__len__.return_value // effective_batch_size
    total_training_steps = steps_per_epoch * train.EPOCHS
    expected_warmup_steps = int(total_training_steps * 0.05)

    mock_training_args.assert_called_once_with(
        output_dir=train.OUTPUT_DIR,
        per_device_train_batch_size=train.BATCH_SIZE,
        gradient_accumulation_steps=train.GRAD_ACCUM,
        num_train_epochs=train.EPOCHS,
        learning_rate=train.LEARNING_RATE,
        weight_decay=train.WEIGHT_DECAY,
        fp16=not mock_torch_cuda_is_bf16_supported.return_value,
        bf16=mock_torch_cuda_is_bf16_supported.return_value,
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        lr_scheduler_type=train.LR_SCHEDULER_TYPE,
        seed=expected_seed,
        eval_strategy="epoch",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        warmup_steps=expected_warmup_steps,
    )

    # SFTTrainer
    mock_early_stopping_callback.assert_called_once_with(
        early_stopping_patience=train.EARLY_STOPPING_PATIENCE
    )
    mock_sft_trainer.assert_called_once_with(
        model=mock_peft_model_instance,  # The PEFT model
        tokenizer=mock_tokenizer_instance,
        train_dataset=mock_train_ds,
        eval_dataset=mock_eval_ds,
        dataset_text_field="text",
        args=mock_training_args.return_value,  # The TrainingArguments instance
        callbacks=[mock_early_stopping_cb_instance],
    )
    mock_trainer_instance.train.assert_called_once()

    # Save model
    mock_peft_model_instance.save_pretrained.assert_called_once_with(train.OUTPUT_DIR)
    mock_tokenizer_instance.save_pretrained.assert_called_once_with(train.OUTPUT_DIR)

    # Memory cleanup
    # train.gc.collect() is called, which is mocked by mock_cuda_seed
    mock_train_gc_collect.assert_called_once()
    # This will also now depend on the patched value of torch.cuda.is_available
    # mock_gc_collect (torch.cuda.is_available) is called again before empty_cache
    mock_torch_cuda_empty_cache.assert_called()  # Assumes CUDA is available as per patch

    # Check os.environ was set
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"

    # Check logging calls (example)
    mock_logger.info.assert_any_call("--- Training Script Started ---")
    mock_logger.info.assert_any_call("Training complete.")
    mock_logger.info.assert_any_call("LoRA adapters and tokenizer saved.")
    mock_logger.info.assert_any_call("Memory freed. Training script complete.")
