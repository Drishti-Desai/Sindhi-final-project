"""Models.

LetterCNN       small ResNet: 64x64 binary letter image -> 52 classes.
WordRecognizer  the same ResNet trunk (initialised from the trained LetterCNN) turns a word
                image into a sequence of feature vectors in reading order, and a Transformer
                decoder emits the Unicode characters one by one.

Why RTL: Sindhi is read right to left, so the image is flipped horizontally before the encoder.
The feature sequence then runs in reading order and the decoder's cross-attention moves
monotonically, which makes training easier. (The flip happens in the dataset, see words.py.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class Trunk(nn.Module):
    """Stem + four stages. `strides` per stage are (h, w) so the recognizer can keep width."""

    def __init__(self, widths=(32, 64, 128, 256), strides=((1, 1), (2, 2), (2, 2), (2, 2))):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for w, s in zip(widths, strides):
            layers.append(nn.Sequential(BasicBlock(cin, w, s), BasicBlock(w, w, 1)))
            cin = w
        self.stages = nn.Sequential(*layers)
        self.out_channels = widths[-1]

    def forward(self, x):
        return self.stages(self.stem(x))


class LetterCNN(nn.Module):
    """`aux` = sizes of optional auxiliary heads (body shape, dot position, dot count).
    They only add training signal; the 52-way head is the model's output."""

    def __init__(self, n_classes: int = 52, widths=(32, 64, 128, 256), dropout: float = 0.3,
                 aux: tuple[int, ...] = ()):
        super().__init__()
        self.trunk = Trunk(widths)
        c = self.trunk.out_channels
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))
        self.aux_heads = nn.ModuleList(nn.Linear(c, n) for n in aux)

    def forward(self, x):
        return self.head(self.trunk(x))

    def forward_all(self, x):
        f = self.trunk(x)
        pooled = self.head[:3](f)
        return self.head[3](pooled), [h(pooled) for h in self.aux_heads]

    def features(self, x):
        """256-d letter embedding (used in Step 5 for letter-level outputs)."""
        return self.head[:2](self.trunk(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):  # x: (B, T, d)
        return x + self.pe[:, : x.size(1)]


class WordRecognizer(nn.Module):
    """CNN encoder + Transformer decoder. Height and width are divided by 8 (same strides as the
    LetterCNN, so its trunk transfers exactly); a 12-letter word of ~400 px gives ~50 time steps."""

    def __init__(self, vocab_size: int, widths=(32, 64, 128, 256), d_model=256, enc_layers=2,
                 dec_layers=3, heads=4, ff=1024, dropout=0.1, max_len=32, height: int = 64):
        super().__init__()
        self.trunk = Trunk(widths)  # identical to LetterCNN's trunk
        c = self.trunk.out_channels
        self.collapse = nn.Conv2d(c, d_model, kernel_size=(height // 8, 1))  # all rows -> 1
        self.enc_pos = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True),
            enc_layers, enable_nested_tensor=False)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dec_pos = PositionalEncoding(d_model, max_len + 2)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True),
            dec_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.checkpointing = False

    def run_trunk(self, images):
        """With self.checkpointing (training only) each stage keeps just its input and recomputes
        its inside during backward: same result, far less memory (full-resolution early layers
        need several GB per batch otherwise on an 8 GB Mac)."""
        if not (self.checkpointing and self.training and images.requires_grad is False and torch.is_grad_enabled()):
            return self.trunk(images)
        from torch.utils.checkpoint import checkpoint
        x = checkpoint(self.trunk.stem, images, use_reentrant=False)
        for stage in self.trunk.stages:
            x = checkpoint(stage, x, use_reentrant=False)
        return x

    def encode(self, images, widths=None):
        f = self.collapse(self.run_trunk(images)).squeeze(2).transpose(1, 2)  # (B, T, d)
        mask = None
        if widths is not None:  # ignore padded columns
            t = f.size(1)
            valid = torch.ceil(widths.float() / 8).clamp(max=t).long()
            mask = torch.arange(t, device=f.device)[None, :] >= valid[:, None]
        return self.encoder(self.enc_pos(f), src_key_padding_mask=mask), mask

    def decode(self, memory, mem_mask, tokens):
        t = tokens.size(1)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=tokens.device), 1)
        x = self.dec_pos(self.embed(tokens) * math.sqrt(self.d_model))
        h = self.decoder(x, memory, tgt_mask=causal, memory_key_padding_mask=mem_mask)
        return self.out(h)

    def forward(self, images, widths, tokens_in):
        memory, mask = self.encode(images, widths)
        return self.decode(memory, mask, tokens_in)

    @torch.no_grad()
    def greedy(self, images, widths, bos: int, eos: int, max_len: int = 32):
        memory, mask = self.encode(images, widths)
        b = images.size(0)
        seq = torch.full((b, 1), bos, dtype=torch.long, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        probs = []
        for _ in range(max_len):
            logits = self.decode(memory, mask, seq)[:, -1]
            p = logits.softmax(-1)
            nxt = p.argmax(-1)
            probs.append(p.max(-1).values)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return seq[:, 1:], torch.stack(probs, 1)


def load_trunk_from_letter_cnn(recognizer: WordRecognizer, letter_state: dict) -> int:
    """Copy the trained LetterCNN trunk weights into the recognizer (shapes match; only strides differ)."""
    own = recognizer.trunk.state_dict()
    copied = {k[len("trunk."):]: v for k, v in letter_state.items()
              if k.startswith("trunk.") and own.get(k[len("trunk."):], None) is not None
              and own[k[len("trunk."):]].shape == v.shape}
    own.update(copied)
    recognizer.trunk.load_state_dict(own)
    return len(copied)
