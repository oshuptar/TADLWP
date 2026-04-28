"""Device, corpus loading, train/val split, and checkpoint save/load."""

import torch

from language_model import GPTConfig, GPTLanguageModel
from tokenizer import encode_decode_from_bpe, load_text_and_tokenize


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: {device} ({torch.cuda.device_count()} GPU(s) visible)")
    else:
        device = "cpu"
        print("Device: cpu (no CUDA GPU detected)")
    return device


def split_data(data, val_ratio=0.1):
    n = int((1 - val_ratio) * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(
        "[split_data] "
        f"Splits: train {len(train_data):,} tokens, val {len(val_data):,} tokens"
    )
    return train_data, val_data


def save_model(model, config, vocab, checkpoint_path, bpe_merges=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "device": config.device,
        "vocab": vocab,
        "bpe_merges": bpe_merges,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"[save_model] Model saved to {checkpoint_path}")


def load_model(checkpoint_path, device=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = device or ckpt.get("device", "cpu")
    config = GPTConfig(
        vocab_size=ckpt["vocab_size"],
        block_size=ckpt["block_size"],
        n_embd=ckpt["n_embd"],
        n_head=ckpt["n_head"],
        n_layer=ckpt["n_layer"],
        device=device,
    )
    model = GPTLanguageModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    vocab = ckpt["vocab"]
    merges = ckpt.get("bpe_merges") or []
    encode, decode = encode_decode_from_bpe(vocab, merges)
    return model, config, encode, decode
