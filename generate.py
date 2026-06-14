"""
Text generation with a trained MiniGPT checkpoint.

Usage:
    python generate.py --prompt "Once upon a time" --tokens 150 --temperature 0.8
"""

import argparse
import torch
from pathlib import Path
import re

from model import MiniGPT
from tokenizer import WordTokenizer
from config import TrainingConfig


def cleanup_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()

    # Merge simple contractions: i ' m -> i'm, don ' t -> don't
    s = re.sub(r"\b(\w+)\s+'\s+(\w+)\b", r"\1'\2", s)

    # Remove spaces before punctuation: "word ." -> "word."
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)

    # Fix bracket spacing: "( hello" -> "(hello", "hello )" -> "hello)"
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)

    # Collapse repeated punctuation tokens
    s = re.sub(r"(?:\s*\.\s*){3,}", "...", s)
    s = re.sub(r"(?:\s*,\s*){2,}", ",", s)
    s = re.sub(r"\.{2,}", "...", s)

    # Fix awkward combined punctuation: "!." -> "!", "?." -> "?"
    s = re.sub(r"([!?])\.", r"\1", s)
    s = re.sub(r"\.\s*([!?])", r".\1", s)

    # Collapse 3+ repeated words: "the the the" -> "the the"
    s = re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1 \1", s, flags=re.IGNORECASE)

    # Remove template noise commonly leaked by small instruction-tuned models
    s = re.sub(r"\b(user|assistant)\s*:\s*", "", s, flags=re.IGNORECASE)

    return s


def load_model_and_tokenizer(checkpoint_path: str, vocab_path: str, device: torch.device):
    """Load model weights and tokenizer from disk."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    # Reconstruct config
    cfg = TrainingConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Load tokenizer
    tokenizer = WordTokenizer(max_vocab_size=cfg.vocab_size)
    tokenizer.load(vocab_path)

    # Build model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_seq_len=cfg.seq_len,
        dropout=0.0,  # no dropout at inference
    ).to(device)

    # Support both full checkpoints (dict with "model_state") and raw state_dicts (e.g. model_ft.pt)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"[Generate] Loaded model from epoch {ckpt.get('epoch', '?')} "
              f"(val_loss={ckpt.get('loss', '?'):.4f})")
    else:
        model.load_state_dict(ckpt)
        print("[Generate] Loaded model state_dict")
    model.eval()
    return model, tokenizer


def generate_text(
    model: MiniGPT,
    tokenizer: WordTokenizer,
    prompt: str,
    max_new_tokens: int = 25,
    temperature: float = 0.3,
    top_k: int = 20,
    top_p: float = 0.8,
    repetition_penalty: float = 1.35,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text continuation from a prompt string."""
    ids = tokenizer.encode(prompt)
    if not ids:
        print("[Warning] Prompt encoded to empty token list. Check vocabulary.")
        return ""

    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Keep outputs short (small models degrade quickly on long generations)
    max_new_tokens = min(int(max_new_tokens), 25)

    output_ids = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Decode only the new tokens (after the prompt).
    # IMPORTANT: stop at the first "<eos>" token BEFORE decoding to a final string.
    new_ids = output_ids[0, len(ids):].tolist()
    tokens = [tokenizer.id2token.get(int(i), "<UNK>") for i in new_ids]
    if "<eos>" in tokens:
        tokens = tokens[: tokens.index("<eos>")]
    generated = " ".join(tokens)

    return cleanup_text(generated)


def interactive_session(model, tokenizer, device, args):
    """Run an interactive REPL for text generation."""
    print("\n" + "═" * 60)
    print("  MiniGPT — Interactive Generation")
    print("  Type a prompt and press Enter. Ctrl-C to quit.")
    print("═" * 60)

    while True:
        try:
            prompt = input("\n> Prompt: ").strip()
            if not prompt:
                continue

            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )

            print(f"\n{'─'*50}")
            print(f"  {cleanup_text(prompt)} {generated}")
            print(f"{'─'*50}")

        except KeyboardInterrupt:
            print("\n\nBye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--checkpoint", default="checkpoints/model_best.pt")
    parser.add_argument("--vocab", default="checkpoints/vocab.json")
    parser.add_argument("--prompt", default=None, help="Starting text (omit for interactive mode)")
    parser.add_argument("--tokens", type=int, default=25, help="Number of new tokens to generate (max 25 recommended)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (0.5=focused, 1.5=creative)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Nucleus sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.35,
                        help="Penalize repeating tokens (1.0=off, 1.3–1.4 recommended)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive REPL")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.checkpoint).exists():
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("Train the model first: python train.py")
        exit(1)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.vocab, device)

    if args.interactive or args.prompt is None:
        interactive_session(model, tokenizer, device, args)
    else:
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(f"\nPrompt : {args.prompt}")
        print(f"Output : {args.prompt} {generated}")
