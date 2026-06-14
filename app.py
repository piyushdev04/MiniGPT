"""
MiniGPT Gradio Playground

- 3-column minimal UI: Controls | Chat Output | Insights
- Streaming token-by-token generation (yields)
- Token-level insights (top-k probabilities per step)
- Basic training monitor tab (live losses + sample generation)

Run:
  python app.py
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import torch
import torch.nn.functional as F

from config import TrainingConfig
from dataset import load_text_file, make_dataloaders
from model import MiniGPT
from tokenizer import WordTokenizer


# -----------------------------
#  Text cleanup (readability)
# -----------------------------

def cleanup_text(s: str) -> str:
    import re

    s = re.sub(r"\s+", " ", (s or "")).strip()

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
    return s


# -----------------------------
#  Theme / CSS (minimal)
# -----------------------------

ACCENT = "#5B7CFF"  # single accent color (blue-ish)

CSS = f"""
:root {{
  --accent: {ACCENT};
}}

/* Typography + whitespace */
.gradio-container {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}}

/* Card styling */
.minicard {{
  border-radius: 16px !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06) !important;
}}

/* Primary button */
.btn-primary button {{
  background: var(--accent) !important;
  border: none !important;
}}

/* Subtle animation */
* {{
  transition: background-color 120ms ease, border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}}

/* Chat tweaks */
div[data-testid="chatbot"] {{
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,0.08);
}}

/* Reduce clutter */
footer {{
  display: none !important;
}}
"""


# -----------------------------
#  Model / tokenizer loading
# -----------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_cfg_from_checkpoint(ckpt: Dict[str, Any]) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg_dict = ckpt.get("config", {}) or {}
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model_and_tokenizer(
    checkpoint_path: str,
    vocab_path: str,
    device: torch.device,
) -> Tuple[MiniGPT, WordTokenizer, TrainingConfig, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = _load_cfg_from_checkpoint(ckpt)

    tokenizer = WordTokenizer(max_vocab_size=cfg.vocab_size)
    tokenizer.load(vocab_path)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_seq_len=cfg.seq_len,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, cfg, ckpt


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
#  Sampling + token insights
# -----------------------------

def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0:
        return logits
    seen = set(generated_ids)
    for token_id in seen:
        if logits[token_id] < 0:
            logits[token_id] *= repetition_penalty
        else:
            logits[token_id] /= repetition_penalty
    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    # logits: (vocab,)
    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > float(top_p)
        remove[..., 0] = False
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    logits = logits / max(float(temperature), 1e-8)
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return int(next_id)


def get_top_token_probs(
    logits: torch.Tensor,
    tokenizer: WordTokenizer,
    n: int = 5,
) -> List[Tuple[str, float]]:
    probs = F.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k=min(int(n), probs.numel()))
    out: List[Tuple[str, float]] = []
    for p, token_id in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.id2token.get(int(token_id), "<UNK>")
        out.append((tok, float(p)))
    return out


def stream_generate(
    model: MiniGPT,
    tokenizer: WordTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    eos_marker: str = "<eos>",
    device: torch.device,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields dict updates:
      - text_so_far: full decoded text (prompt + generated)
      - new_token: last generated token str
      - top5: list[(token, prob)] for this step
      - generated_tokens: list[str]
      - generated_ids: list[int]
    """
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        yield {
            "text_so_far": prompt,
            "new_token": "",
            "top5": [],
            "generated_tokens": [],
            "generated_ids": [],
        }
        return

    generated: List[int] = list(prompt_ids)
    generated_tokens: List[str] = []

    model.eval()
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            context = generated[-model.max_seq_len :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            next_logits = logits[0, -1, :].float().clone()

            next_logits = apply_repetition_penalty(next_logits, generated, float(repetition_penalty))
            top5 = get_top_token_probs(next_logits, tokenizer, n=5)
            next_id = sample_next_token(next_logits, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p))

            generated.append(next_id)
            tok = tokenizer.id2token.get(int(next_id), "<UNK>")
            generated_tokens.append(tok)

            # Stop on explicit end marker (added by downloader).
            if tok.strip().lower() == eos_marker.strip().lower():
                break

            text_so_far = tokenizer.decode(generated, skip_special=True)
            yield {
                "text_so_far": text_so_far,
                "new_token": tok,
                "top5": top5,
                "generated_tokens": list(generated_tokens),
                "generated_ids": list(generated),
            }


# -----------------------------
#  Chat formatting
# -----------------------------

def build_chat_prompt(history: List[Dict[str, str]], user_message: str, max_turns: int = 12) -> str:
    """
    Minimal, extensible prompt format.
    Accepts Gradio "messages" format: [{"role": "user"|"assistant", "content": "..."}].
    Keeps last N turns (approx), concatenates with lightweight role tags.
    """
    # max_turns counts user+assistant pairs; we keep roughly 2*max_turns messages.
    msgs = history[-(max_turns * 2) :] if max_turns else history
    parts: List[str] = []
    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append(f"User: {user_message}\nAssistant:")
    return "\n\n".join(parts).strip()


# -----------------------------
#  Training monitor (streaming)
# -----------------------------

def training_stream(
    *,
    cfg: TrainingConfig,
    device: torch.device,
) -> Generator[Dict[str, Any], None, None]:
    """
    Minimal streaming trainer for UI monitoring.
    Yields:
      - step_text
      - train_loss_history
      - val_loss_history
      - last_sample
      - status
    """
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    status = "Loading data..."
    yield {
        "status": status,
        "step_text": "",
        "train_loss_history": [],
        "val_loss_history": [],
        "last_sample": "",
    }

    text = load_text_file(cfg.data_path, max_chars=cfg.max_chars)
    tokenizer = WordTokenizer(max_vocab_size=cfg.vocab_size)
    tokenizer.build_vocab(text)
    token_ids = tokenizer.encode(text)
    train_loader, val_loader = make_dataloaders(
        token_ids,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_seq_len=cfg.seq_len,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))

    def evaluate() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total += float(loss.item())
        return total / max(len(val_loader), 1)

    train_hist: List[float] = []
    val_hist: List[float] = []

    best_val = float("inf")
    last_sample = ""

    status = "Training..."
    t0 = time.time()

    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())

            if (step + 1) % int(cfg.log_every) == 0:
                avg = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                step_text = f"epoch={epoch+1}/{cfg.epochs} step={step+1}/{len(train_loader)} loss={avg:.4f} lr={lr:.2e} time={elapsed:.0f}s"
                yield {
                    "status": status,
                    "step_text": step_text,
                    "train_loss_history": list(train_hist),
                    "val_loss_history": list(val_hist),
                    "last_sample": last_sample,
                }

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss = evaluate()
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss

        # Sample generation from first prompt
        prompt = (cfg.sample_prompts[0] if cfg.sample_prompts else "once upon a time").strip()
        ids = tokenizer.encode(prompt)
        x0 = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x0, max_new_tokens=40, temperature=0.8, top_k=40, top_p=0.95)
        last_sample = tokenizer.decode(out[0].tolist(), skip_special=True)

        step_text = (
            f"epoch={epoch+1}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} (ppl={math.exp(min(train_loss, 20)):.1f}) "
            f"val_loss={val_loss:.4f} (ppl={math.exp(min(val_loss, 20)):.1f})"
        )
        yield {
            "status": status,
            "step_text": step_text,
            "train_loss_history": list(train_hist),
            "val_loss_history": list(val_hist),
            "last_sample": last_sample,
        }

    status = f"Done. Best val_loss={best_val:.4f}"
    yield {
        "status": status,
        "step_text": "",
        "train_loss_history": list(train_hist),
        "val_loss_history": list(val_hist),
        "last_sample": last_sample,
    }


def plot_losses(train_losses: List[float], val_losses: List[float]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.2, 3.2), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Loss (per epoch)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if train_losses:
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="train", linewidth=2)
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="val", linewidth=2)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
#  Gradio app
# -----------------------------

def build_app() -> gr.Blocks:
    device = get_device()

    default_checkpoint = "checkpoints/model_best.pt"
    default_vocab = "checkpoints/vocab.json"

    # Note: Some Gradio versions expect css/theme on launch(), not Blocks().
    # We keep Blocks() minimal for compatibility and apply styling at launch.
    with gr.Blocks() as demo:
        # IMPORTANT (Gradio 6.x):
        # State components must be created INSIDE the Blocks context,
        # otherwise event preprocessing can crash with KeyError on State IDs.
        model_state = gr.State(value=None)      # type: ignore[arg-type]
        tok_state = gr.State(value=None)        # type: ignore[arg-type]
        cfg_state = gr.State(value=None)        # type: ignore[arg-type]
        ckpt_state = gr.State(value=None)       # type: ignore[arg-type]
        # Gradio 6.12 Chatbot expects "messages" format:
        # [{"role": "user"|"assistant", "content": "..."}]
        chat_history_state = gr.State(value=[])  # list[dict]

        gr.Markdown(
            "### MiniGPT Playground\n"
            "Minimal UI for generation, chat, and token insights.",
        )

        with gr.Tabs():
            with gr.TabItem("Playground", id="play"):
                with gr.Row(equal_height=True):
                    # ---------------- Left panel ----------------
                    with gr.Column(scale=3, min_width=300):
                        with gr.Group(elem_classes=["minicard"]):
                            gr.Markdown("**Controls**")
                            prompt_in = gr.Textbox(
                                label="Prompt",
                                lines=8,
                                placeholder="Write a prompt…",
                            )
                            chat_mode = gr.Checkbox(label="Chat mode", value=True)

                            temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
                            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
                            max_tokens = gr.Slider(1, 512, value=120, step=1, label="Max new tokens")
                            repetition_penalty = gr.Slider(1.0, 1.5, value=1.2, step=0.01, label="Repetition penalty")

                            with gr.Row():
                                generate_btn = gr.Button("Generate", elem_classes=["btn-primary"])
                                clear_btn = gr.Button("Clear")

                            status_md = gr.Markdown("", elem_id="status")

                        with gr.Group(elem_classes=["minicard"]):
                            gr.Markdown("**Model / Files**")
                            checkpoint_path = gr.Textbox(label="Checkpoint", value=default_checkpoint)
                            vocab_path = gr.Textbox(label="Vocab", value=default_vocab)
                            load_btn = gr.Button("Load / Reload")

                    # ---------------- Center panel ----------------
                    with gr.Column(scale=6, min_width=520):
                        with gr.Group(elem_classes=["minicard"]):
                            gr.Markdown("**Conversation**")
                            # Keep Chatbot args compatible across Gradio versions
                            chatbot = gr.Chatbot(label="", height=520)
                            with gr.Row():
                                clear_chat_btn = gr.Button("Clear conversation")

                    # ---------------- Right panel ----------------
                    with gr.Column(scale=3, min_width=320):
                        with gr.Group(elem_classes=["minicard"]):
                            gr.Markdown("**Insights**")
                            stats_md = gr.Markdown("")
                            tokens_box = gr.Textbox(label="Generated tokens (latest run)", lines=8, interactive=False)
                            top5_md = gr.Markdown("")
                            attention_placeholder = gr.Markdown(
                                "<div style='opacity:0.7'>Attention visualization: <i>placeholder</i></div>"
                            )

            with gr.TabItem("Training", id="train"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4, min_width=360):
                        with gr.Group(elem_classes=["minicard"]):
                            gr.Markdown("**Training controls**")
                            data_path = gr.Textbox(label="Data path", value=TrainingConfig().data_path)
                            epochs = gr.Slider(1, 50, value=TrainingConfig().epochs, step=1, label="Epochs")
                            batch_size = gr.Slider(1, 256, value=TrainingConfig().batch_size, step=1, label="Batch size")
                            lr = gr.Number(label="Learning rate", value=TrainingConfig().learning_rate)
                            seq_len = gr.Slider(16, 512, value=TrainingConfig().seq_len, step=1, label="Seq length")
                            vocab_size = gr.Slider(1000, 50000, value=TrainingConfig().vocab_size, step=100, label="Vocab size cap")
                            start_train_btn = gr.Button("Start training", elem_classes=["btn-primary"])
                            train_status = gr.Markdown("")
                            train_step = gr.Markdown("")

                    with gr.Column(scale=6, min_width=520):
                        with gr.Group(elem_classes=["minicard"]):
                            gr.Markdown("**Monitoring**")
                            loss_plot = gr.Plot()
                            last_sample_box = gr.Textbox(label="Last sample", lines=10, interactive=False)

        # -----------------------------
        #  UI helper functions
        # -----------------------------

        def _ensure_loaded(
            checkpoint: str,
            vocab: str,
            cur_model,
            cur_tok,
            cur_cfg,
            cur_ckpt,
        ):
            ckpt_ok = Path(checkpoint).exists()
            vocab_ok = Path(vocab).exists()
            if not ckpt_ok or not vocab_ok:
                msg = []
                if not ckpt_ok:
                    msg.append(f"Missing checkpoint: `{checkpoint}`")
                if not vocab_ok:
                    msg.append(f"Missing vocab: `{vocab}`")
                return None, None, None, None, "\n".join(msg)

            model, tok, cfg, ckpt = load_model_and_tokenizer(checkpoint, vocab, device=device)
            n_params = count_parameters(model)
            stats = (
                f"**Device**: `{device.type}`\n\n"
                f"**Vocab size**: `{tok.vocab_size}`\n\n"
                f"**Seq length**: `{cfg.seq_len}`\n\n"
                f"**Parameters**: `{n_params:,}`\n\n"
                f"**Config**: `{type(cfg).__name__}`"
            )
            return model, tok, cfg, ckpt, stats

        def load_click(checkpoint: str, vocab: str, cur_model, cur_tok, cur_cfg, cur_ckpt):
            model, tok, cfg, ckpt, stats = _ensure_loaded(checkpoint, vocab, cur_model, cur_tok, cur_cfg, cur_ckpt)
            status = "" if model is not None else stats
            return model, tok, cfg, ckpt, stats, status

        load_btn.click(
            load_click,
            inputs=[checkpoint_path, vocab_path, model_state, tok_state, cfg_state, ckpt_state],
            outputs=[model_state, tok_state, cfg_state, ckpt_state, stats_md, status_md],
            queue=True,
        )

        # Auto-load on startup (best-effort)
        demo.load(
            load_click,
            inputs=[checkpoint_path, vocab_path, model_state, tok_state, cfg_state, ckpt_state],
            outputs=[model_state, tok_state, cfg_state, ckpt_state, stats_md, status_md],
            queue=True,
        )

        def clear_all():
            return "", [], "", "", ""

        clear_btn.click(
            clear_all,
            inputs=[],
            outputs=[prompt_in, chatbot, tokens_box, top5_md, status_md],
            queue=False,
        )

        def clear_chat():
            return [], []

        clear_chat_btn.click(
            clear_chat,
            inputs=[],
            outputs=[chatbot, chat_history_state],
            queue=False,
        )

        def _format_top5(top5: List[Tuple[str, float]]) -> str:
            if not top5:
                return ""
            lines = ["**Top token probabilities (latest step)**", ""]
            for tok, p in top5:
                lines.append(f"- `{tok}`  —  **{p:.3f}**")
            return "\n".join(lines)

        def generate_click(
            prompt: str,
            is_chat: bool,
            temp: float,
            k: int,
            p: float,
            max_new: int,
            rep_pen: float,
            checkpoint: str,
            vocab: str,
            cur_model,
            cur_tok,
            cur_cfg,
            cur_ckpt,
            chat_hist: List[Dict[str, str]],
        ):
            # Ensure model is loaded
            model, tok, cfg, ckpt, stats = _ensure_loaded(checkpoint, vocab, cur_model, cur_tok, cur_cfg, cur_ckpt)
            if model is None or tok is None or cfg is None:
                yield (
                    cur_model,
                    cur_tok,
                    cur_cfg,
                    cur_ckpt,
                    chat_hist,
                    chat_hist,
                    "",
                    "",
                    stats,
                    "",
                    gr.update(interactive=True),
                )
                return

            prompt = (prompt or "").strip()
            if not prompt:
                yield (
                    model,
                    tok,
                    cfg,
                    ckpt,
                    chat_hist,
                    chat_hist,
                    "",
                    "",
                    stats,
                    "",
                    gr.update(interactive=True),
                )
                return

            # Disable while running + show thinking indicator
            yield (
                model,
                tok,
                cfg,
                ckpt,
                chat_hist,
                chat_hist,
                "",
                "",
                stats,
                "MiniGPT is thinking...",
                gr.update(interactive=False),
            )

            if is_chat:
                chat_hist = list(chat_hist)
                chat_hist.append({"role": "user", "content": prompt})
                chat_hist.append({"role": "assistant", "content": ""})
                ui_chat = list(chat_hist)
                full_prompt = build_chat_prompt(chat_hist[:-2], prompt)
            else:
                # Single-shot mode: show as one-turn conversation (messages format)
                ui_chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
                # Keep prompt consistent with the instruction dataset format.
                full_prompt = f"User: {prompt}\nAssistant:"

            generated_tokens: List[str] = []
            last_top5: List[Tuple[str, float]] = []

            # Stream token-by-token
            for upd in stream_generate(
                model,
                tok,
                full_prompt,
                max_new_tokens=int(max_new),
                temperature=float(temp),
                top_k=int(k),
                top_p=float(p),
                repetition_penalty=float(rep_pen),
                eos_marker="<eos>",
                device=device,
            ):
                # Extract only newly generated part for display
                # We keep UI minimal: only update assistant message with new decoded text.
                full_text = upd["text_so_far"]
                last_top5 = upd["top5"]
                generated_tokens = upd["generated_tokens"]

                # Decode assistant response as "everything after the last 'Assistant:'" for chat.
                if is_chat:
                    if "Assistant:" in full_text:
                        assistant_text = full_text.split("Assistant:", 1)[-1].strip()
                    else:
                        assistant_text = full_text
                    # Stop if the model starts producing a new user turn (template leakage)
                    if "\nUser:" in assistant_text:
                        assistant_text = assistant_text.split("\nUser:", 1)[0].strip()
                    ui_chat[-1]["content"] = assistant_text
                    chat_hist[-1]["content"] = assistant_text
                else:
                    # In single mode, assistant is the full continuation (prompt + generation),
                    # but we keep it clean by showing only the generated continuation.
                    base_ids = tok.encode(prompt)
                    all_ids = tok.encode(full_text)
                    cont_ids = all_ids[len(base_ids) :] if len(all_ids) >= len(base_ids) else all_ids
                    assistant_text = tok.decode(cont_ids, skip_special=True).strip()
                    ui_chat[-1]["content"] = assistant_text

                # Clean up spacing/punctuation for readability
                ui_chat[-1]["content"] = cleanup_text(ui_chat[-1]["content"])
                if is_chat:
                    chat_hist[-1]["content"] = ui_chat[-1]["content"]

                yield (
                    model,
                    tok,
                    cfg,
                    ckpt,
                    ui_chat,
                    chat_hist,
                    cleanup_text(" ".join([t for t in generated_tokens[-250:] if t.strip().lower() != "<eos>"])),
                    _format_top5(last_top5),
                    stats,
                    "",
                    gr.update(interactive=False),
                )

            # Re-enable at end
            yield (
                model,
                tok,
                cfg,
                ckpt,
                ui_chat,
                chat_hist,
                cleanup_text(" ".join([t for t in generated_tokens[-250:] if t.strip().lower() != "<eos>"])),
                _format_top5(last_top5),
                stats,
                "",
                gr.update(interactive=True),
            )

        generate_btn.click(
            generate_click,
            inputs=[
                prompt_in,
                chat_mode,
                temperature,
                top_k,
                top_p,
                max_tokens,
                repetition_penalty,
                checkpoint_path,
                vocab_path,
                model_state,
                tok_state,
                cfg_state,
                ckpt_state,
                chat_history_state,
            ],
            outputs=[
                model_state,
                tok_state,
                cfg_state,
                ckpt_state,
                chatbot,
                chat_history_state,
                tokens_box,
                top5_md,
                stats_md,
                status_md,
                generate_btn,
            ],
            queue=True,
        )

        # -----------------------------
        #  Training tab wiring
        # -----------------------------

        train_hist_state = gr.State(value=[])  # train losses
        val_hist_state = gr.State(value=[])    # val losses

        def start_training_click(
            data: str,
            n_epochs: int,
            bs: int,
            lr_val: float,
            seq: int,
            vocab_cap: int,
        ):
            cfg = TrainingConfig()
            cfg.data_path = data
            cfg.epochs = int(n_epochs)
            cfg.batch_size = int(bs)
            cfg.learning_rate = float(lr_val)
            cfg.seq_len = int(seq)
            cfg.vocab_size = int(vocab_cap)

            for upd in training_stream(cfg=cfg, device=device):
                train_losses = upd["train_loss_history"]
                val_losses = upd["val_loss_history"]
                fig = plot_losses(train_losses, val_losses) if (train_losses or val_losses) else None
                yield (
                    upd["status"],
                    upd["step_text"],
                    train_losses,
                    val_losses,
                    fig,
                    upd["last_sample"],
                )

        start_train_btn.click(
            start_training_click,
            inputs=[data_path, epochs, batch_size, lr, seq_len, vocab_size],
            outputs=[train_status, train_step, train_hist_state, val_hist_state, loss_plot, last_sample_box],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=16)
    # Compatibility: different Gradio versions accept different launch kwargs.
    try:
        app.launch(css=CSS, theme=gr.themes.Soft())
    except TypeError:
        try:
            app.launch(css=CSS)
        except TypeError:
            app.launch()

