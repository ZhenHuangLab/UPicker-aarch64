# UPicker — Linux aarch64 + CUDA 13 + GB10 (sm_121)

Upstream: https://github.com/JachyLikeCoding/UPicker
(*Briefings in Bioinformatics* 2024, doi:10.1093/bib/bbae636)

## Result: 🔴 blocked **upstream**, not by the architecture

The table recorded "freeze.yml pins extremely specific versions, Linux aarch64
'does not exist'". That is true but it is the smaller problem. The published
repository is **missing six source modules** and cannot run on any platform.

### The environment file is unusable anywhere

`freeze.yml` is a raw `conda env export` from the author's machine: every
package carries an exact build string, including `blas=1.0=mkl` (x86-only by
construction) and `cudatoolkit=11.6.0`. `requirements_upicker.txt` is the same
snapshot as pip pins — `torch==1.12.1`, `torchvision==0.13.1`, `numpy==1.23.5`,
`timm==0.9.7`. Neither can resolve on linux-aarch64; the `mkl` pin alone makes
the conda file unsolvable off x86.

Replacing it with the standard recipe works fine:

```bash
mamba create -y -p ./env -c conda-forge python=3.11 numpy scipy matplotlib pillow opencv
./env/bin/pip install torch torchvision      # torch 2.13.0+cu130 aarch64
```

### Six modules are absent from the repository

A relative-import audit of the published tree:

| importing file | missing module |
|---|---|
| `models/upicker/deformable_transformer.py` | `models.upicker.ops` (whole package) |
| `models/upicker/backbone.py` | `models.upicker.convnext` |
| `models/upicker/backbone.py` | `models.upicker.swin_transformer` |
| `models/upicker/upicker.py` | `models.registry` |
| `models/upicker/upicker.py` | `models.upicker.segmentation` |
| `datasets/__init__.py`, `datasets/coco.py` | `datasets.torchvision_datasets` |

`import models.upicker` fails immediately with
`ModuleNotFoundError: No module named 'models.upicker.convnext'`. The README
documents only `conda env create -f freeze.yml` and never mentions building or
fetching anything else, so this is an incomplete publication rather than a
missing build step.

## What was reconstructed here: `models/upicker/ops`

The `ops` package is the one piece that is both essential and unambiguous — it
is Deformable-DETR's multi-scale deformable attention, which UPicker uses
verbatim (`from .ops.modules import MSDeformAttn`). It is normally a hand-written
CUDA extension; it has been reimplemented here in **pure PyTorch** using the
`F.grid_sample` formulation that Deformable-DETR ships as its own reference
kernel:

```text
models/upicker/ops/
├── functions/ms_deform_attn_func.py   ms_deform_attn_core_pytorch()
└── modules/ms_deform_attn.py          MSDeformAttn
```

This runs on CPU and CUDA with no compilation, so there is no per-architecture
or per-PyTorch-version build to maintain.

Verified on GB10 (torch 2.13.0+cu130):

| check | result |
|---|---|
| forward, cpu | `(2, 50, 64)`, all finite |
| forward, cuda | `(2, 50, 64)`, all finite |
| backward | input gradient and all parameter gradients finite |
| 2-d `reference_points` path | OK |
| 4-d `reference_points` path | OK |

## Verdict

Reconstructing the other five modules would mean re-deriving unpublished code
from DINO / Deformable-DETR / ConvNeXt / Swin upstreams, and the result could
not honestly be called "UPicker". **Recommendation: ask the authors for the
complete tree**; nothing about aarch64 is blocking. Every aarch64-relevant piece
(PyTorch, the deformable-attention kernel) is verified working here.
