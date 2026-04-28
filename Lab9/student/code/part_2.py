"""Part 2: training loop and TrainResult."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from tokenizer import create_lm_dataloaders
from language_model import GPTConfig, GPTLanguageModel
from part_1 import avg_loss_on_loader


@dataclass
class TrainResult:
    train_loss: float
    val_loss: float
    total_iters: int


def train(model, config : GPTConfig, train_loader, val_loader):
    print(f"[train] Training ({config.epochs} epochs)...")
    device = config.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = 0
    last_train_loss = 0.0
    last_val_loss = 0.0

    """
    TODO: Train with AdamW for config.epochs, log train/val loss, return TrainResult.
    """
    model.to(device)
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        n_tokens = 0
        for x,y in train_loader:
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)
            logits = model(x)
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
            loss.backward()
            optimizer.step()
            n = B * T
            running_loss += loss.item() * n
            n_tokens += n
            total_steps += 1

        train_loss_epoch = running_loss / n_tokens
        val_loss_epoch = avg_loss_on_loader(model, val_loader, device, config.eval_iters)
        last_train_loss, last_val_loss = train_loss_epoch, val_loss_epoch

        if epoch % config.eval_interval == 0 or epoch == config.epochs:
            print(f"[train] Epoch {epoch}: train loss {train_loss_epoch}")

    return TrainResult(last_train_loss, last_val_loss, total_steps)



# ------ Tests ------


def test_train_result_fields():
    r = TrainResult(train_loss=1.5, val_loss=2.0, total_iters=100)
    assert r.train_loss == 1.5 and r.val_loss == 2.0 and r.total_iters == 100
    print("  test_train_result_fields [OK]")


def _make_tiny_loaders():
    vocab_size = 50
    data = torch.randint(0, vocab_size, (120,))
    n = len(data) // 2
    train_data, val_data = data[:n], data[n:]
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=8,
        n_embd=32,
        n_head=2,
        n_layer=1,
        epochs=4,
        eval_interval=2,
        eval_iters=2,
        batch_size=52,
        device="cpu",
    )
    train_loader, val_loader = create_lm_dataloaders(train_data, val_data, config.block_size, config.batch_size)
    model = GPTLanguageModel(config)
    return model, config, train_loader, val_loader


def test_train_returns_train_result():
    model, config, train_loader, val_loader = _make_tiny_loaders()
    result = train(model, config, train_loader, val_loader)
    assert isinstance(result, TrainResult)
    assert result.total_iters == 4
    print("  test_train_returns_train_result [OK]")


def test_train_losses_finite():
    model, config, train_loader, val_loader = _make_tiny_loaders()
    result = train(model, config, train_loader, val_loader)
    assert result.train_loss == result.train_loss
    assert result.val_loss == result.val_loss
    print("  test_train_losses_finite [OK]")


def main():
    print("=== Part 2: training loop ===\n")
    test_train_result_fields()
    test_train_returns_train_result()
    test_train_losses_finite()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
