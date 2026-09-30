# neuronloop project status

## Classification

Research scratch — a personal, unpublished experiment directory. Not a library,
not a tool, not a package: 32 Jupyter notebooks plus the local `nn.py` fork of
the author's compositional layer library (published separately as `cznn`) and
local `rev.py` reversible-network helpers.

## Status

Archive / abandoned. Written 2024-12-03 → 2026-02-13, never published, never
finished. The founding idea — a "neuron loop" with per-neuron gates — is present
in `neuronloop.ipynb` only, in two cells, and is then abandoned for generic
attention/convolution architecture search. There is no runnable entrypoint, no
saved outputs (except two embedded figures), no reproducibility, and no
conclusion. Kept as provenance for the `cznn` `neuronloop` fork, not as a
working project.

## Evidence

- `neuronloop.ipynb` — the origin: `NeuronLoop(GRU)` emits per-neuron logits `n`
  and gates per-neuron reconstruction error (`n[...,0]`) plus a next-step
  coupling term (`n[...,1]`); two `Loop` implementations then iterate a shared
  block N times with a softmax over iteration logits. One training curve survives
  (loss 3.5 → 2.6, "acc" 0.09 → ~0.25). Later cells reference undefined names,
  so the notebook is not runnable as-is.
- `Untitled-Copy5.ipynb` — the largest single sweep (~651 code lines, 16
  experiment cells) over `GroupAttention` / `GroupConvAttention` variants on the
  synthetic sorting task; its embedded figure shows one variant at ~0.35
  accuracy and the rest at ~0.15–0.20.
- `Untitled-Copy4/6/7/8/9/10/11/12/13`, `neuronloop-Copy1/2` — near-duplicate
  iterations of the same family (`Attention`, `Bitention`, `GroupAttention`,
  `GroupBitention`, `ConvAttention`, `Attentions`/`Bitentions` path toggles),
  almost always benchmarked on `x = torch.rand(b, l); x.argsort()` with
  cross-entropy.
- `Untitled-Copy3.ipynb`, `Untitled-Copy4.ipynb` — the only surviving numeric
  results, pasted as markdown `loss:acc` tables (~1.1–1.7 / 60–72 % for a
  char-level LM; ~5.4–5.5 / 0.4–3.6 % for the sorting sweeps).
- `Untitled.ipynb`, `Untitled-Copy1/2.ipynb` — char-level Chinese LM / denoising
  experiments (noise-probability schedule, sampled text) over the absent
  `../datbrick` corpora; `Untitled4.ipynb` — n-gram entropy of `cn*.B/.H`;
  `qa.ipynb` — a chat-QA model sketch over an absent `../chatgpt` JSON.
- Reversible networks — `rev.py` / `rev.old.py` / `nn.old.py` and
  `Untitled5/6*.ipynb`:
  reversible residual pairs, FFT and least-squares inverse convolution, memory
  tests.
- `Untitled3.ipynb` — a utility cell that strips `outputs` from a notebook,
  which is why almost no results are preserved.
- `nn.py` — verified body-identical to `cznn/variants_neuronloop.py` (only a
  module docstring differs); imports cleanly under torch 2.14 and exposes all
  ten fork-specific symbols.

## Boundaries

- Nothing runs end-to-end. `neuronloop.ipynb` references undefined names; most
  notebooks hard-code `.cuda()` and read `../datbrick` / `../chatgpt` corpora
  that are not in this repo.
- Almost all experimental results were deliberately discarded
  (`Untitled3.ipynb`), and the two surviving figures have no underlying data or
  logs.
- No tests, no CI, no packaging, no environment pin, no checkpoints, no license.
- `Untitled1.ipynb` was an employer work artifact (license-clearing dump) and
  was redacted before publication; `Untitled8/9` had employer/spreadsheet paths
  genericised. See the README "Data governance" section.
- `nn.py` is a divergent fork of the library now published as `cznn`; it is kept
  here verbatim so the notebooks' original `from nn import *` works. Do not
  merge it with the canonical fork or rewrite the notebook imports.
