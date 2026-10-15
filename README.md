# neuronloop

**What:** A personal research scratch directory (2024–2026, all timestamps
2024-12-03 / 2026-02-13) that started from a "neuron loop" idea — a small
recurrent block whose *per-neuron* and *per-iteration* gates decide how far each
element keeps iterating — and then drifted into a broader architecture search
over attention/convolution hybrids composed with the author's `nn.py` layer
library.

**Why it exists / what problem it was attacking:** whether iterative,
per-neuron-gated computation can substitute for (or augment) stacked attention
on sequence tasks, and, once that question was open, a fast place to compose and
benchmark attention/convolution variants. The honest summary: the "neuron loop"
itself is one notebook and two experiments; everything after it is a sandbox of
near-duplicate `Untitled-CopyN` architecture-search sessions, not a coherent
project with a thesis, a dataset, or a conclusion.

## How to run

Notebook-only. **There is no single entrypoint** — no package, no CLI, no
`main`, no tests, no CI. Start with **`neuronloop.ipynb`** (see *Canonical
notebook* below), then the sorting-task sweeps in `Untitled-Copy5.ipynb` /
`Untitled-Copy8.ipynb`.

```bash
pip install torch numpy matplotlib          # core (torch 2.14 tested)
pip install pandas scipy jieba zhconv einops ipywidgets tqdm  # extras
jupyter lab
```

- Run notebooks **from this directory** — they do `from nn import *` and
  `from rev import *`, and `nn.old.py` / `rev.old.py` are kept as the prior
  versions some cells reload.
- Most training cells hard-code `.cuda()`; there is no CPU fallback.
- The files they read (`../datbrick/cn*.txt|.B|.H`, `../datbrick/py.*`,
  `../chatgpt/qa/backend/qa.json`) are **not** in this repo, so the real-data
  notebooks cannot run. The synthetic-task notebooks generate
  `torch.rand(...)` on the fly and *can* run (GPU aside).
- Results were not kept: nearly every notebook's outputs are empty, and
  `Untitled3.ipynb` is a utility cell that deliberately strips `outputs` from
  another notebook.

## What is here vs not

**Here (36 files):**

- **32 notebooks**, ~5.9k non-blank code lines (7.1k before the redaction below).
- `nn.py` (174 lines) — the `neuronloop` fork of the layer library (see
  *Relationship to `cznn`*), required by **14** of the notebooks.
- `rev.py` (195 lines) — reversible-network helpers, used by 3 notebooks, plus
  `nn.old.py` (262 lines) and `rev.old.py` (182 lines), the previous versions.
- Exactly **two results survived** as embedded figures: the training curve in
  `neuronloop.ipynb` (loss ~3.5 → 2.6, "acc" 0.09 → ~0.25, OneCycle LR decay) and
  an accuracy comparison in `Untitled-Copy5.ipynb` (one `GroupAttention` variant
  reaches ~0.35 while the rest plateau near 0.15–0.20). Two more results survive
  as pasted `loss:acc` text tables in markdown cells of `Untitled-Copy3.ipynb`
  and `Untitled-Copy4.ipynb` (~1.1–1.7 loss / 60–72% acc on a char LM).

**Not here:**

- The `../datbrick` text corpora and the `../chatgpt/qa` JSON — so no real-data
  experiment is reproducible from this repo.
- Any checkpoints, logs, or trained weights; any environment pin/lockfile; any
  tests.
- The saved outputs of almost every notebook (only the two figures above).
- One employer work artifact — see *Data governance*.

## Layout

| group | notebooks | what they do |
|---|---|---|
| loop origin | `neuronloop.ipynb`, `neuronloop-Copy1.ipynb`, `neuronloop-Copy2.ipynb` | `NeuronLoop(GRU)` per-neuron gate; `Loop` (run a block N times, softmax over iteration logits, weighted output); then attention on synthetic sorting |
| synthetic attention search (bulk) | `Untitled-Copy4..10`, `Untitled11`, `Untitled12`, `Untitled13` | `Attention` / `Bitention` / `GroupAttention` / `GroupBitention` / `ConvAttention` variants; task is almost always `x = rand(b,l); x.argsort()` → cross-entropy on the permutation (a few use smooth-L1 regression) |
| char-LM / denoising | `Untitled.ipynb`, `Untitled-Copy1.ipynb`, `Untitled-Copy2.ipynb`, `Untitled-Copy3.ipynb` | char-level Chinese/Python LM with a noise-probability schedule (`m[0]["p"] = …`) and sampled text; needs the missing `../datbrick` corpora |
| corpus statistics | `Untitled4.ipynb` | n-gram entropy of memmapped `cn*.B/.H` corpora |
| reversible nets / invertible conv | `Untitled5.ipynb`, `Untitled6.ipynb`, `Untitled6 - Copy.ipynb`, `Untitled6-Copy1.ipynb`, `Untitled6-Copy2.ipynb`, `rev.py` | reversible residual pairs, FFT/least-squares inverse convolution, memory experiments |
| work / scratch | `Untitled1.ipynb` (redacted), `Untitled8.ipynb` (genericised), `Untitled9.ipynb`, `Untitled2/7/10/14.ipynb`, `qa.ipynb` | see *Data governance* and *Boundaries* |

**Canonical notebook: `neuronloop.ipynb`.** It is the only file that contains the
project's original idea and the only complete training run with a preserved
figure. It is still not clean: later cells are order-dependent and reference
names (`Pos2`, `dbg`) defined nowhere in the file, so it does not run
start-to-finish as-is. `Untitled-Copy5.ipynb` is the largest and most complete
*single sweep* (651 code lines, 16 experiment cells), but it has no narrative.

## Relationship to `cznn`

The local `nn.py` is the **`neuronloop` fork of the author's compositional layer
library**, which is now published as
[`github.com/ZisIsNotZis/cznn`](https://github.com/ZisIsNotZis/cznn). It is kept
there **verbatim** as `cznn/variants_neuronloop.py` (a one-line module docstring
is prepended; the rest of the file is byte-identical to this one). It is also
kept **here**, unchanged, because the notebooks `from nn import *` — the import
path is the notebooks' original one and has deliberately not been rewritten.

The `neuronloop` fork and the canonical `retnet` fork share only five symbols
(`Seq`, `Add`, `Mul`, `Res`, `F`). This fork's distinctive additions are `Map`,
`Const`, `Weight`, `Bias`, `Cat`, `Einsum`, `SwiGLU`, `Attention`,
`MultiheadAttention`, and `Nn`. The forks are kept side by side rather than
merged because merging them is a real refactor with name collisions (two
different `SwiGLU`s). See the `cznn` README for the full comparison.

`rev.py` / `rev.old.py` have no published counterpart; they are local.

## Data governance / redactions

A handful of files were publication-sanitised on 2026-09-30 (the source archive
`/home/z/my/neuronloop` is untouched):

- `Untitled1.ipynb` — **redacted**. It was not research at all but a leftover
  employer work artifact: an open-source license-clearing dump (project
  codename, clearing states, full third-party component list). Its contents were
  removed; the remaining cells are the original analysis stubs.
- `Untitled8.ipynb` — the employer spreadsheet path/name and a column header
  were replaced with generic placeholders (`data/tags.xlsx`, `tag`); the toy
  co-occurrence → 2-D embedding experiment itself is unchanged.
- `Untitled9.ipynb`, `Untitled4.ipynb`, `Untitled6 - Copy.ipynb` — an absolute
  Windows user path and absolute drive paths (`C:\Users\<user>\…`, `d:/…`,
  `d:/my/…`) were genericised to placeholder/relative paths.

No `/home/<user>/...` data paths existed in the code; all corpus references were
already relative (`../datbrick/...`) and were left as-is.

## Status

**Archive / research scratch, 2024-12-03 → 2026-02-13. Abandoned, incomplete,
not a coherent project.** The unifying idea (per-neuron / per-iteration gating
in a loop) appears in one notebook and is dropped after two cells; the bulk is a
pile of near-duplicate `Untitled-CopyN` architecture sweeps on synthetic sorting
data with no saved results and no conclusion. Published only so the `cznn`
`neuronloop` fork has its originating context and so the small amount of real
evidence (two figures, two loss tables) is visible. Nothing here runs
end-to-end.

**License:** AGPL-3.0-only; see [`LICENSE`](LICENSE). This is personal
research code, not a redistribution-ready package.
