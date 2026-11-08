# PinPoint3D interactive annotation workspace

This workspace adapts interSeg3D-Studio to PinPoint3D's native positive and
negative point prompts. Upstream sources and model weights are not committed;
`bootstrap.sh` clones the pinned revisions and applies the local Studio patch.
The checkpoint is downloaded separately from the official PinPoint3D source.

**Single-user session:** The backend stores the point cloud and annotation state
globally. Use one browser or test at a time: parallel requests can overwrite
each other's state and exported results. Bind the services to localhost.

## Setup

Tested on Linux with an RTX 2070 (8 GB, sm75), CUDA 11.8, and Chrome. Install
`pixi`, `just`, Git, Node.js and npm first. From this directory:

```bash
just bootstrap         # clone pinned upstreams and apply the Studio patch
just install           # isolated locked Pixi environment
just minkowski         # compile MinkowskiEngine for sm75; requires CUDA driver
just frontend-install
just ui-test-install
just sample-ply        # convert the host repository's ScanNet sample
```

Download the pretrained checkpoint linked in the [PinPoint3D README](https://github.com/Quit123/PinPoint3D)
to `weights/pinpoint3d_pretrained.pth`. The tested file is 174,971,280 bytes and
has SHA-256 `0794bbba07c15277235dba747b6994a1d88678ecd93ca6a39f428171def3eace`.
Connect it to Studio without copying it:

```bash
mkdir -p interSeg3D-Studio/src/backend/pinpoint3d/weights
ln -s "$(pwd)/weights/pinpoint3d_pretrained.pth" \
  interSeg3D-Studio/src/backend/pinpoint3d/weights/checkpoint1099.pth
```

The MinkowskiEngine build is a source-built exception to the Pixi lock. The
build recipe defaults to CUDA architecture 7.5; other GPUs may need a different
architecture and have not been validated here.

## Use and verify

Run `just backend` and `just frontend` in separate terminals, then open
`http://127.0.0.1:3001`. Upload `outputs/ply/scene0025_00.ply`, create an object,
place an Object (positive) point and, where useful, a Background (negative)
point, then run segmentation and save results. Segmentation and export need no
API key; external object recognition is optional.

With both services running, `just ui-test` exercises the real model through the
browser and checks the exported PLY and metadata. `just test-metadata` checks
export IDs, colours and click flags. `just validate-backend` separately tests
the real HTTP inference and export path. `just run-all` runs the saved prompts
against the bundled ScanNet scene; `just figures` renders the comparison.
See [ui-tests/README.md](ui-tests/README.md) for browser test details. Generated
images, logs, checkpoints and environments remain local and are ignored by Git.

On the tested chair example, a positive click followed by a negative click on
nearby clutter reduced the browser mask from 2,234 to 1,933 points. A separate
raw-model comparison against the dataset mask gave IoU 0.801 versus 0.909;
ground truth helped select those prompts, so this is a single assisted example,
not a general accuracy estimate.

## Upstream revisions

- [PinPoint3D](https://github.com/Quit123/PinPoint3D): `40cb81ec81bb9b525009abbee0f727d92d80db44`
- [interSeg3D-Studio](https://github.com/zh-plus/interSeg3D-Studio): `4605802f547aa17e67e58e4331cb6e64c2bb0bf7`
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine): `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`

The Studio patch is maintained here rather than as a published fork. Consult
the upstream projects' terms before redistributing their source or weights.
