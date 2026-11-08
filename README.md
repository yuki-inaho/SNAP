<div align="center">
<h1>SNAP: Towards Segmenting Anything in Any Point Cloud (3DV 2026 Best Paper Candidate)</h1>
<a href="https://arxiv.org/abs/2510.11565"><img src="https://img.shields.io/badge/arXiv-2509.16909-b31b1b" alt="arXiv"></a>
<a href="https://neu-vi.github.io/SNAP/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

<a href="https://aniket-gupta1.github.io/">Aniket Gupta<sup>1</sup></a>
&nbsp;&nbsp;
<a href="https://sarihust.github.io/">Hanhui Wang<sup>1</sup></a>
&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/charlie-saunders-82b091179/">Charles Saunders<sup>2</sup></a>
&nbsp;&nbsp;
<a href="https://arunirc.github.io/">Aruni RoyChowdhury<sup>2</sup></a>
&nbsp;&nbsp;
<a href="https://coe.northeastern.edu/people/singh-hanumant/">Hanumant Singh<sup>1</sup></a>
&nbsp;&nbsp;
<a href="https://jianghz.me/">Huaizu Jiang<sup>1</sup></a>
<br>
<sup>1</sup> Northeastern University &nbsp;&nbsp;
<sup>2</sup> The Mathworks, Inc.

</div>

This repo is the official project repository for the paper **SNAP: Towards Segmenting Anything in Any Point Cloud**. SNAP is the first model which supports interactive segmentation of point clouds from any domain (indoor, outdoor and aerial). We provide multiple model checkpoints trained on different domains as well as the full model trained on all datasets from different domains.

<p align="center">
      <img src="assets/SNAP_small.gif" alt="Demo" width="700">
</p>

<!-- <table>
  <tr>
    <td>
      <img src="assets/Performance_radar_chart.png" width="405" alt="Performance of SNAP on different domains">
    </td>
    <td width="405">
    </td>
  </tr>
</table> -->

## Updates
* [Dec 6, 2025] Code released.
* [Nov 5, 2025] SNAP is accepted for Oral presentation at 3DV 2026.
* [Oct 13, 2025] Preprint released.

For a separate interactive workflow with native positive and negative prompts,
see the [PinPoint3D annotation workspace](pinpoint3d-workspace/README.md).


## Installation

### Pixi environment (recommended)

The repository ships a locked [Pixi](https://pixi.sh) workspace (`pixi.toml` +
`pixi.lock`). It pins Python 3.10, PyTorch 2.4.1 with CUDA 12.4, the matching
PyG extensions, spconv, CLIP, Open3D, PyVista/VTK and PyQt6, so the CUDA and
PyTorch-extension ABI stay consistent.

Requirements:
- Linux x86_64 with an NVIDIA GPU and driver (validated on RTX 2070 / sm75,
  driver 580.178.04 reporting CUDA 13.0; only the CUDA 12.4 *runtime* is used)
- [Pixi](https://pixi.sh) 0.79 or newer
- [just](https://github.com/casey/just) is optional but convenient

```shell
git clone <repo-url> SNAP
cd SNAP
pixi install --locked
```

Verify the environment:

```shell
pixi run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# or, with just:
just gpu          # Python / PyTorch / CUDA / GPU versions
just imports      # import model, dataset and GUI modules
just validate     # CUDA matmul, torch-scatter, torch-cluster and a spconv kernel
just model-smoke  # build SNAP on the GPU (attention backend chosen automatically)
```

Run commands inside the environment with `pixi run <command>`, or open a shell
with `pixi shell`. The justfile wraps the common workflows:

```shell
just            # list recipes
just install    # pixi install --locked
just test       # non-GUI pytest
just test-gui   # point picking and segmentation UI logic
just test-gui-qt # real Qt/VTK widgets under Xvfb
just test-all   # all three suites, in separate processes
just test-attention # attention backend and chunking regression checks
just verify-inference # pretrained GPU inference on a labelled ScanNet sample
just benchmark-precision # compare FP32 and mixed FP16 on the same prompts
just demo data_examples/ScanNet/scene0011_00 Indoor
```

`just test` runs the non-GUI tests, `just test-gui` checks the GUI interaction
logic, and `just test-gui-qt` exercises actual Qt/VTK widgets using `xvfb-run`.
Install Xvfb on the host to run the latter. `just test-all` runs all three in
separate processes to isolate Qt/VTK and CUDA state.

### GPU compatibility and FlashAttention-2

SNAP's backbone can use FlashAttention-2 on Ampere/Ada/Hopper (sm80+).
FlashAttention-2 does **not** support Turing (sm75, e.g. RTX 2070), so
`flash-attn` is deliberately **not** installed by default. The model selects
its attention backend automatically:

```shell
# auto (default): FlashAttention-2 when available, otherwise PyTorch attention
# flash: require FlashAttention-2 and fail loudly if it is unavailable
# normal: force the PyTorch attention fallback
export SNAP_ATTENTION_BACKEND=auto   # or flash / normal
```

On an sm80+ GPU you can install the optional build (it is not part of the
default lock):

```shell
just install-flash
```

The recipe refuses to run on Turing and points at the fallback instead. The
fallback computes attention in patches and can use more memory than FlashAttention;
the full SNAP-C Outdoor demo (KITTI) ran on the 8 GB RTX 2070 with about 1.5 GB
peak allocated CUDA memory. FlashAttention and the fallback have not been
compared on this machine, which cannot run the former.

The fallback limits the size of each attention chunk to 256 MiB by default.
For lower peak memory, set the budget in bytes before launching the demo:

```shell
SNAP_ATTN_CHUNK_BYTES=67108864 just demo data_examples/ScanNet/scene0011_00 Indoor
```

The limit applies to an attention tensor, not the whole model's memory usage.

### Opt-in FP16 inference

FP16 is opt-in; FP32 remains the default for both entry points.

```shell
# Interactive GUI (default is fp32)
pixi run python iterative_gui_demo.py --pc_path data_examples/ScanNet/scene0011_00 \
    --domain Indoor --resume checkpoints/SNAP_C.pth --precision fp16

# Same through the justfile: the fifth positional argument is the precision.
just demo data_examples/ScanNet/scene0011_00 Indoor 0.02 checkpoints/SNAP_C.pth fp16

# Recommended memory setting on the 8 GB RTX 2070: FP16 plus a 64 MiB
# attention budget.
just demo-lowmem data_examples/ScanNet/scene0011_00 Indoor 0.02
```

```python
from python_demo import SegmentationModel

model = SegmentationModel(
    checkpoint_path="checkpoints/SNAP_C.pth",
    domain="Indoor",
    grid_size=0.02,
    precision="fp16",  # default: "fp32"
)
```

`precision="fp16"` (CLI: `--precision fp16`) enables CUDA autocast around the
backbone, the mask decoder and the CLIP text encoder. It does **not** call
`model.half()`: model parameters stay FP32, and coordinate/feature
preprocessing, prompt handling, text matching, mask/IoU post-processing and
cached text tokens are kept in FP32. `AutocastSafeSubMConv3d` keeps sparse
submanifold convolutions in FP32 because spconv 2.3.6 has no FP16 algorithm for
some of them on Turing. FP16 is rejected with a clear error on CPU, other
non-CUDA devices and CUDA devices below compute capability 5.3; use FP32 there.

Measured with the real `SNAP_C.pth`, ScanNet `scene0011_00`, grid size 0.02 and
one prompt on the RTX 2070 (8 GB, sm75, PyTorch attention fallback). The GUI
row includes model construction and the initial backbone pass; the API row is
the backbone/segment path after the model is loaded.

| setting | GUI startup + backbone peak | GUI click-flow peak | API backbone peak |
|---|---:|---:|---:|
| FP32, 64 MiB chunk | 1706.9 MiB | 1422.7 MiB | 1706.9 MiB |
| FP16, 64 MiB chunk | 1393.3 MiB | 1384.3 MiB | 1393.3 MiB |
| FP32, default chunk (256 MiB) | 1915.5 MiB | 1422.7 MiB | - |
| FP16, default chunk (256 MiB) | 1962.9 MiB | 1384.3 MiB | - |

The smaller attention budget was required for the memory reduction here: with
the default chunk FP16 was slightly higher than FP32 in this run. A prior
model-only benchmark on the same scene and three prompts measured
1368.2 MiB (FP32) versus 1063.4 MiB (FP16) with the 64 MiB chunk, and
1576.7 MiB versus 1610.8 MiB with the default chunk. All reported runs produced
finite, non-empty masks and kept every parameter in FP32. The GUI runs were
driven by a real Qt right-click on a cloud point.

These are single-scene, single-GPU measurements, not a general quality claim.
The same API prompt produced 6389 mask points with FP16 and 6902 with FP32;
mean mask IoU against FP32 was 0.9961 over three prompts in the prior benchmark.
Validate FP16 on the target data before relying on it. The interactive demo
still defaults to FP32.


### PyTorch extension compatibility

`torch-scatter 2.1.2` and `torch-cluster 1.6.3` come from the `pyg` channel and
are built against `torch 2.4.0 + cu124`. They are ABI-compatible with the pinned
`torch 2.4.1 + cu124`, but they are **not** compatible with a different torch or
CUDA major/minor line. If you change `pytorch` in `pixi.toml`, update the PyG
builds at the same time, then re-run:

```shell
just lock
just install
just validate
```

`spconv-cu120 2.3.6` is a prebuilt wheel that includes sm70/sm75/sm80/sm86/sm89
kernels, so it works on Turing as well as newer GPUs. `numpy` is pinned below 2
because the older model and dataset code expects the NumPy 1.x ABI.

### Legacy conda environment

`environment.yaml` is kept for reference and for reproducing the original
paper environment. It is not required for the Pixi workflow above.

```shell
conda env create -f environment.yaml
conda activate Lidar_Seg
```

The old barebones instructions installed `flash-attn` unconditionally and
therefore fail on Turing GPUs; use the Pixi environment and the attention
fallback instead.

## Interactive Demo
Download the SNAP-C checkpoint from [here](#checkpoints) into a “checkpoints/” folder in the current working directory.
```shell
pixi run python iterative_gui_demo.py --pc_path <path to point cloud files> --domain <domain> --resume <path to model checkpoint>
```

Examples for the provided point cloud files:
```shell
pixi run python iterative_gui_demo.py --pc_path data_examples/S3DIS/conferenceRoom_1/ --domain Indoor --resume checkpoints/SNAP_C.pth
pixi run python iterative_gui_demo.py --pc_path data_examples/ScanNet/scene0011_00 --domain Indoor --resume checkpoints/SNAP_C.pth
pixi run python iterative_gui_demo.py --pc_path data_examples/KITTI/000000.bin  --domain Outdoor --resume checkpoints/SNAP_C.pth
pixi run python iterative_gui_demo.py --pc_path data_examples/nuScenes/example_3.pcd.bin --domain=Outdoor --resume checkpoints/SNAP_C.pth
```

The same demo through the justfile:
```shell
just demo data_examples/KITTI/000000.bin Outdoor 0.05
just demo data_examples/ScanNet/scene0011_00 Indoor 0.02
```

Once the demo is launched, it'll load the pointcloud in a visualization window, you can segment objects using following steps:
1. Left click to rotate the point cloud, middle click to pan, middle click scroll to zoom in/out and <b>right click to mark prompt points</b>.
2. Right click on a object to segment it. You can mark multiple points on the same object to improve segmentation.
3. Press "New object" button to segment a new object.
4. Press "Finalize Segment" button to finalize the segmentation.

If you want to visualize the results on the complete data (if you have the full dataset), set the data_root parameter in the config file of the respective dataset and run 
```shell
pixi run torchrun --nproc_per_node=1 vis_all.py --distributed --use_pdnorm --resume checkpoints/SNAP_C.pth --stage scannet --iterative --enable_amp
```

## Python Snippets to run the model
Run the snippet inside the environment (`pixi run python` or `pixi shell`):
```shell
# Import the required libraries
from python_demo import PointCloudData, SegmentationModel

# Load the point cloud data
point_cloud = PointCloudData()
point_cloud_data = point_cloud.load_point_cloud("data_examples/KITTI/000000.bin")

# Load the model
model = SegmentationModel(checkpoint_path="checkpoints/SNAP_C.pth", domain="Outdoor", grid_size=0.05)

# Intialize the model and run the model backbone to extract point features
point_cloud_data = model.intialize_pointcloud(point_cloud_data)
model.extract_backbone_features(point_cloud_data)

# Specify a prompt point
# prompt_points = [[[x1, y1, z1], [x2, y2, z2], ...]]  # List of prompt pointe, Shape -> M, P, 3 (M = number of objects, P = number of clicks on each object)
# The prompt points should be in the format [x, y, z] and should be in the same coordinate system as the point cloud data
prompt_points = [[[ 9.4854517 ,  7.34119511, -0.40044212]]]

# Run segmentation
masks, text_labels, iou_scores = model.segment(point_cloud_data, prompt_points, text_prompt=None)

# Visualize the results
model.visualize_results(point_cloud_data, masks, text_labels, iou_scores)

```

Note- Outdoor scenes use a grid size of 0.05 and Indoor scenes use a grid size of 0.02.

## Checkpoints
We provide the following checkpoints - 
#### Dataset Models
1. [SNAP - KITTI](https://drive.google.com/file/d/1LGlLWjK9Kz__3WDBWst4PxgUVc44MSga/view?usp=drive_link) - Model trained only on SemanticKITTI dataset
2. [SNAP - ScanNet](https://drive.google.com/file/d/1jym_zdTJlrdEmkZJ4VXDufw4iMmXkjPk/view?usp=drive_link) - Model trained only on ScanNet dataset

#### Domain Models
1. [SNAP - Indoor](https://drive.google.com/file/d/1Sm4J6S9CJxgCF2Rm1hXS16e74CDigYN1/view?usp=drive_link) - Model trained on ScanNet + HM3D
2. [SNAP - Outdoor](https://drive.google.com/file/d/128PoQvnmupWmWWLWNfzHyyQrv2nUpukq/view?usp=drive_link) - Model trained on SemanticKITTI + nuScenes + Pandaset
3. [SNAP - Aerial](https://drive.google.com/file/d/1VnRuXuLfxs-CTG5_bAbJ5MO7Kd-MSCAK/view?usp=drive_link) - Model trained on STPLS3D + DALES

#### Full Models
1. [SNAP - C ](https://drive.google.com/file/d/1xGzyOxUPhXLRO8rdOWX9NOetAdBg9q0w/view?usp=drive_link) - Model trained on ScanNet + HM3D + SemanticKITTI + nuScenes + Pandaset + STPLS3D + DALES (Recommended for general use)
2. [SNAP - C (without HM3D)](https://drive.google.com/file/d/10MXoZMGUhLBCrsYWtZNbxehHpqTw58km/view?usp=drive_link) - - Model trained on ScanNet + SemanticKITTI + nuScenes + Pandaset + STPLS3D + DALES. 
3. [SNAP - C (without HM3D + variable grid sizes)](https://drive.google.com/file/d/1OSlVecpMOHmjecII1JZTJdzr0TJY1tdY/view?usp=drive_link) - - Model trained on ScanNet + HM3D + SemanticKITTI + nuScenes + Pandaset + STPLS3D + DALES along with variable grid sizes data augmentation. (Recommended for use when using different voxelization sizes than the default ones used.)

## Training
### Data
* **Outdoor**:- [SemanticKITTI](https://semantic-kitti.org/), [Nuscenes](https://www.nuscenes.org/), 
[Pandaset](https://scale.com/open-av-datasets/pandaset)
* **Indoor**:- [ScanNet200](https://rozdavid.github.io/scannet200), [HM3D](https://niessner.github.io/HM3D/)
* **Aerial**:- [DALES](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php), [STPLS3D](https://www.stpls3d.com/)

### Scripts
```shell
pixi run torchrun --nproc_per_node=1 main.py --distributed --use_pdnorm --resume <path to checkpoint to resume from> --stage <datasets you want to train on> --iterative --enable_amp --use_localized_loss --use_aux_loss --use_random_clicks --use_pdnorm
```

or with `just` (arguments are forwarded unchanged):
```shell
just train --distributed --use_pdnorm --stage scannet --iterative --enable_amp
```

## Evaluation
```shell
pixi run torchrun --nproc_per_node=1 main.py --exp_name <experiment_name> --stage <dataset_name> --compute_NOC --enable_amp --use_pdnorm --distributed --batch_size=1 --num_workers=8 --num_prompt_points=20 --num_object_chunk=16 --iterative --use_centroid --val_only --resume checkpoints/SNAP_C.pth
```

or with `just` (adds `--val_only` automatically):
```shell
just eval --exp_name <experiment_name> --stage <dataset_name> --compute_NOC --enable_amp --use_pdnorm --distributed --batch_size=1 --num_workers=8 --num_prompt_points=20 --num_object_chunk=16 --iterative --use_centroid --resume checkpoints/SNAP_C.pth
```

Note that use_centroid flag enables the use of centroid prompts during evaluation and val_only flag runs only the validation loop.


## Citation

```bibtex
@misc{gupta2025snapsegmentingpointcloud,
      title={SNAP: Towards Segmenting Anything in Any Point Cloud}, 
      author={Aniket Gupta and Hanhui Wang and Charles Saunders and Aruni RoyChowdhury and Hanumant Singh and Huaizu Jiang},
      year={2025},
      eprint={2510.11565},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.11565}, 
}
```

## Acknowledgements
We thank the authors of [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) and [Pointcept](https://github.com/Pointcept/Pointcept) for their open-source codebases which were used in parts of this project. 
