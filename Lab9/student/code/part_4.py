"""Part 4: load checkpoint and generate (QA-style ``Question:`` / ``Answer:`` format)."""

from pathlib import Path

import torch

from helpers import load_model


def get_answer(text: str) -> str:
    start = text.find("Answer:")
    if start == -1:
        start = 0
    start += len("Answer:")
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def ask_nano_gpt(
    checkpoint_path,
    prompt,
    max_new_tokens=80,
    device=None,
) -> None:
    """
    TODO: Load checkpoint, loop on input, Question/Answer prompt, generate, print answer.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raise NotImplementedError("Implement ask_nano_gpt.")


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    checkpoint_path = code_dir / "model.pt"
    ask_nano_gpt(checkpoint_path, prompt="", max_new_tokens=80)


if __name__ == "__main__":
    main()