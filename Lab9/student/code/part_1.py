"""Part 1: average loss over a DataLoader (cycles the loader if it is short)."""

from itertools import cycle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokenizer import TokenWindowDataset
from language_model import GPTConfig, GPTLanguageModel


def avg_loss_on_loader(model, loader, device, max_batches: int) -> float:
    """
    TODO: Token-weighted mean loss over up to max_batches batches (cycle the loader if it is short).
    """
    model.eval()
    total_loss : float = 0.0
    weight : int = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(cycle(loader)): # cycles the loader in case it is shorter than the max_batches we require
            if i >= max_batches:
                break
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)
            logits: torch.Tensor = model(x)
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            n = x.numel()
            total_loss += loss.item() * n
            weight += n

    model.train()
    return total_loss / weight if weight else 0.0


def test_avg_loss_on_loader_finite():
    torch.manual_seed(0)
    vocab_size = 20
    data = torch.randint(0, vocab_size, (64,))
    block_size, batch_size = 8, 16
    ds = TokenWindowDataset(data, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=32,
        n_head=2,
        n_layer=1,
        device="cpu",
    )
    model = GPTLanguageModel(cfg)
    loss = avg_loss_on_loader(model, loader, "cpu", max_batches=3)
    assert isinstance(loss, float)
    assert loss == loss
    print("  test_avg_loss_on_loader_finite [OK]")


def main():
    print("=== Part 1: avg_loss_on_loader ===\n")
    test_avg_loss_on_loader_finite()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
