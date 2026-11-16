"""Masked-LM scores for correction candidates (XLM-RoBERTa), and the detector's per-token error probability.

A candidate replaces its span of input tokens (with the span's trailing punctuation) by k <mask> tokens, where k is
the number of pieces the candidate text has; its score is the sum of the log-probabilities of those pieces. All
candidates with the same span and the same k share one forward pass. The output layer is applied only at the mask
positions (the 250k-word vocabulary would otherwise dominate memory).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

from xlmr.correct import Position, is_word


class MaskedScorer:
    def __init__(self, model_dir, device, batch: int = 32):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device).eval()
        self.device, self.batch = device, batch
        self._ids: dict[str, list[int]] = {}

    def ids(self, text: str) -> list[int]:
        if text not in self._ids:
            self._ids[text] = self.tok(text, add_special_tokens=False)["input_ids"]
        return self._ids[text]

    @torch.no_grad()
    def _run(self, seqs: list[list[int]], masks: list[list[int]]) -> list[torch.Tensor]:
        """Log-probabilities over the vocabulary at the given positions of each sequence."""
        out = []
        for b in range(0, len(seqs), self.batch):
            chunk, mk = seqs[b:b + self.batch], masks[b:b + self.batch]
            L = max(len(s) for s in chunk)
            x = torch.full((len(chunk), L), self.tok.pad_token_id)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, s in enumerate(chunk):
                x[i, :len(s)] = torch.tensor(s)
                att[i, :len(s)] = 1
            h = self.model.roberta(input_ids=x.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
            rows = torch.tensor([i for i, m in enumerate(mk) for _ in m], device=self.device)
            cols = torch.tensor([j for m in mk for j in m], device=self.device)
            lp = F.log_softmax(self.model.lm_head(h[rows, cols]).float(), -1).cpu()
            k = 0
            for m in mk:
                out.append(lp[k:k + len(m)])
                k += len(m)
        return out

    def score(self, tokens: list[str], positions: list[Position], only=None) -> None:
        """Fill cand.lm for every candidate of every word position (or only the positions in `only`)."""
        tok_ids = [self.ids(t) for t in tokens]
        jobs, owners = [], []
        for i, p in enumerate(positions):
            if not is_word(p.core) or len(p.cands) < 2 or (only is not None and i not in only):
                continue
            groups: dict[tuple[int, int], list] = {}
            for c in p.cands:
                end = i + c.consumes
                if end > len(tokens):
                    continue
                text = " ".join(c.words) + positions[end - 1].tail
                pieces = self.ids(text)
                groups.setdefault((c.consumes, len(pieces)), []).append((c, pieces))
            for (span, k), members in groups.items():
                left = [t for ids in tok_ids[:i] for t in ids]
                right = [t for ids in tok_ids[i + span:] for t in ids]
                seq = [self.tok.cls_token_id] + left + [self.tok.mask_token_id] * k + right + [self.tok.sep_token_id]
                if len(seq) > 510:   # very long sentence: keep a window around the masks
                    lo = max(0, len(left) - 200)
                    seq = [self.tok.cls_token_id] + left[lo:] + [self.tok.mask_token_id] * k + right[:200] + [self.tok.sep_token_id]
                    left = left[lo:]
                start = 1 + len(left)
                jobs.append((seq, list(range(start, start + k))))
                owners.append(members)
        if not jobs:
            return
        lps = self._run([j[0] for j in jobs], [j[1] for j in jobs])
        for lp, members in zip(lps, owners):
            for c, pieces in members:
                c.lm = float(sum(lp[j, t] for j, t in enumerate(pieces)))


class Detector:
    """P(token is wrong) for each input token, from the fine-tuned token-classification model."""

    def __init__(self, model_dir, device, batch: int = 32):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
        self.device, self.batch = device, batch

    @torch.no_grad()
    def probs(self, sentences: list[list[str]]) -> list[list[float]]:
        out = []
        for b in range(0, len(sentences), self.batch):
            chunk = sentences[b:b + self.batch]
            enc = self.tok(chunk, is_split_into_words=True, truncation=True, max_length=256, padding=True,
                           return_tensors="pt")
            p = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.softmax(-1)[..., 1].cpu()
            for i, words in enumerate(chunk):
                first = {}
                for j, w in enumerate(enc.word_ids(i)):
                    if w is not None and w not in first:
                        first[w] = float(p[i, j])
                out.append([first.get(k, 0.0) for k in range(len(words))])
        return out
