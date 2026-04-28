"""Part 3: corpus → BPE → train → checkpoint (same pipeline as the original ``main.py``)."""

from pathlib import Path

from tokenizer import create_lm_dataloaders
from helpers import get_device, load_text_and_tokenize, save_model, split_data
from language_model import GPTConfig, GPTLanguageModel
from part_2 import TrainResult, train


def train_nano_gpt(
    input_path,
    checkpoint_path,
    val_ratio=0.1,
    epochs=10,
    eval_interval=1,
    eval_iters=50,
    batch_size=32,
    block_size=64,
    n_embd=128,
    n_head=4,
    n_layer=3,
    num_merges=100,
) -> TrainResult:
    """
    TODO: Load text, BPE-tokenize, split, train, save checkpoint with vocab and merges.
    """
    data, vocab, merges = load_text_and_tokenize(input_path, num_merges=num_merges)
    config: GPTConfig = GPTConfig(vocab_size=len(vocab),
                                  block_size=block_size,
                                  n_embd=n_embd,
                                  n_head=n_head,
                                  n_layer=n_layer,
                                  epochs=epochs,
                                  eval_interval=eval_interval,
                                  eval_iters=eval_iters)
    model: GPTLanguageModel = GPTLanguageModel(config)
    train_data, val_data = split_data(data, val_ratio=val_ratio)
    train_loader, val_loader = create_lm_dataloaders(train_data, val_data, block_size, batch_size)
    device = get_device()
    model.to(device)
    train_result: TrainResult = train(model, config, train_loader, val_loader)
    save_model(model, config, vocab, checkpoint_path, merges)
    return train_result
    


# ------ Tests ------


def test_train_nano_gpt_like_main():
    """Same training call as ``main()`` / ``tmp/main.py``; skips if ``input.txt`` is missing."""
    code_dir = Path(__file__).resolve().parent
    input_path = code_dir / "input.txt"
    checkpoint_path = code_dir / "model.pt"
    if not input_path.is_file():
        print(f"  test_train_nano_gpt_like_main [SKIP] (no {input_path.name!r} in {code_dir})")
        return
    r = train_nano_gpt(
        input_path,
        checkpoint_path,
        epochs=3,
        eval_interval=1,
        eval_iters=50,
        batch_size=32,
        block_size=128,
        num_merges=1000,
    )
    assert checkpoint_path.is_file()
    assert isinstance(r, TrainResult)
    assert r.total_iters >= 1
    print("  test_train_nano_gpt_like_main [OK]")


def main() -> None:
    """Train on ``input.txt`` and save ``model.pt`` (same entry behavior as ``tmp/main.py`` training step)."""
    code_dir = Path(__file__).resolve().parent
    input_path = code_dir / "input.txt"
    checkpoint_path = code_dir / "model.pt"

    if not input_path.is_file():
        print(f"Missing {input_path}")
        return

    train_nano_gpt(
        input_path,
        checkpoint_path,
        epochs=3,
        eval_interval=1,
        eval_iters=50,
        batch_size=32,
        block_size=128,
        num_merges=1000,
    )


if __name__ == "__main__":
    main()
