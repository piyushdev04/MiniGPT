"""
demo.py — Run a complete mini training + generation demo without any external data.
Creates a small synthetic corpus, trains for a few epochs, and generates text.

Run: python demo.py
"""

import torch
import math

from model import MiniGPT
from tokenizer import WordTokenizer
from dataset import make_dataloaders
from config import tiny_config


# ── 1. Create a tiny synthetic corpus ────────────────────────────────────── #

SAMPLE_TEXT = """
Once upon a time in a land far away there lived a brave knight named Arthur .
Arthur loved to explore the deep dark forest near his castle .
One day Arthur found a mysterious cave hidden behind a waterfall .
Inside the cave there was a dragon sleeping on a pile of gold coins .
The dragon opened one eye and looked at Arthur with curiosity .
Arthur bowed his head and said I come in peace noble dragon .
The dragon smiled and said I have been waiting for a brave knight like you .
Together they went on many adventures through mountains and valleys .
They crossed raging rivers and climbed steep rocky cliffs .
Arthur and the dragon became the best of friends in all the land .
The king heard of their friendship and invited them to the royal palace .
Everyone celebrated with a great feast that lasted three days and three nights .
The dragon breathed small harmless flames to light the candles on the table .
Arthur told stories of their adventures to all the guests at the feast .
From that day forward peace and happiness spread across the entire kingdom .
Children would wave at the dragon as he flew gracefully through the blue sky .
The old wizard who lived in the tower watched them with a warm smile .
He knew that true friendship was the greatest magic of all .
Once upon a time the world was full of wonder and magic .
Every river had a spirit and every mountain had a name and a story .
The trees whispered secrets to those who listened with a quiet heart .
Stars at night would guide lost travelers safely back to their homes .
A young girl named Elara could hear the songs of the wind .
She would sit by the window and listen to tales of distant lands .
One morning a silver fox appeared at her door with a message .
The message was written in golden ink on a leaf from the oldest tree .
It said come and see what lies beyond the misty mountain range .
Elara packed a small bag and set off on the winding forest path .
She met a talking owl who offered to be her guide and companion .
Together they climbed higher and higher until the air was cold and thin .
At the top they saw a valley filled with glowing lights and music .
People from every corner of the world had gathered there to celebrate .
They were celebrating the return of the lost star that had wandered away .
Elara looked up and saw the star shining brighter than all the others .
The owl told her that her courage had brought the star back home .
She smiled and understood that small acts of bravery change the world .
The king and queen ruled their land with kindness and wisdom .
Every morning the queen walked through the market and greeted everyone .
The king planted trees along the roads so travelers would have shade .
Their children learned to read from the great library in the castle .
Books of every color lined the tall stone walls from floor to ceiling .
The royal cook made bread and soup for anyone who was hungry .
Musicians played in the courtyard every evening at sunset .
The sound of laughter echoed through the cobblestone streets .
A young boy named Tom dreamed of becoming a great explorer one day .
His grandmother told him that the sea held more stories than the sky .
Tom built a small boat from wood he found near the old mill .
He painted it blue and named it after the morning star .
""" * 8   # repeat to get more training data


def demo():
    print("=" * 55)
    print("  MiniGPT — Quick Demo")
    print("=" * 55)

    cfg = tiny_config()
    cfg.epochs = 6
    cfg.log_every = 30

    device = torch.device("cpu")   # demo runs on CPU

    # ── Tokenize ─────────────────────────────────────────
    tokenizer = WordTokenizer(max_vocab_size=cfg.vocab_size)
    tokenizer.build_vocab(SAMPLE_TEXT)
    token_ids = tokenizer.encode(SAMPLE_TEXT)
    print(f"\n[Data] {len(token_ids):,} tokens | vocab size: {tokenizer.vocab_size}")

    # ── DataLoaders ───────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        token_ids, seq_len=cfg.seq_len, batch_size=cfg.batch_size, val_split=0.1
    )

    # ── Model ─────────────────────────────────────────────
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_seq_len=cfg.seq_len,
        dropout=cfg.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {n_params:,} parameters  ({n_params/1000:.1f}K)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # ── Training ──────────────────────────────────────────
    print("\n[Training]")
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                _, loss = model(x, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        ppl = math.exp(min(val_loss, 20))

        print(f"  Epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ppl={ppl:.1f}")

    # ── Generation ────────────────────────────────────────
    print("\n[Generation]")
    prompts = [
        "once upon a time",
        "arthur and the dragon",
        "the king and queen",
        "elara looked up",
    ]

    model.eval()
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                x, max_new_tokens=30, temperature=0.8, top_k=30, top_p=0.9
            )
        new_ids = out[0, len(ids):].tolist()
        generated = tokenizer.decode(new_ids)
        print(f"\n  Prompt : \"{prompt}\"")
        print(f"  Output : {prompt} {generated}")

    print("\n✓ Demo complete! See README.md for training on your own corpus.")


if __name__ == "__main__":
    demo()
