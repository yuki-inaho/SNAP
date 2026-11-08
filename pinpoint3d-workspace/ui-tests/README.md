# interSeg3D-Studio UI tests

Browser-level checks for the interactive frontend. They drive the real UI with
Playwright: upload PLY → create object → positive click → inference → negative
click → inference → mask visibility → save/download.

## Prerequisites

1. Backend listening where the frontend proxies (`VITE_API_BASE_URL`, default
   `http://127.0.0.1:9500`). Either the real PinPoint3D backend or the clearly
   labelled deterministic mock.
2. Frontend dev server on `http://127.0.0.1:3001` (the DEV-only projection hook
   `window.__interSeg3D` is needed by the test).
3. The ScanNet sample PLY. Default path:
   `../outputs/ply/scene0025_00.ply` (generated with `just sample-ply`).

## Run

```bash
# against the deterministic mock (mask must shrink after the negative click)
MOCK_BACKEND=1 \
SAMPLE_PLY=../outputs/ply/scene0025_00.ply \
BASE_URL=http://127.0.0.1:3001 \
npx playwright test

# against the real PinPoint3D backend
MOCK_BACKEND=0 CUBE_SIZE=0.02 npx playwright test

# final real-backend evidence run (screenshots, ZIP and extracted metadata)
BASE_URL=http://127.0.0.1:3001 MOCK_BACKEND=0 CUBE_SIZE=0.02 \
SAMPLE_PLY=../outputs/ply/scene0025_00.ply ARTIFACT_DIR=artifacts/final-real \
npx playwright test
```

Artifacts (screenshots and the downloaded zip) are written to
`artifacts/run/` unless `ARTIFACT_DIR` is set. The test logs
`mask-before-negative`, `mask-after-negative`, `mask-shrink`, `download` and
`export-metadata` as annotations.

The workspace `just ui-test` recipe writes to `artifacts/just-ui-test`.
Artifacts are generated locally and are not checked in.

## Real-export regression assertions

The real-backend run unzips the downloaded package and asserts the metadata
contract that the UI depends on:

- `objects[0]` is `{ id: 1, part_id: 1, mask_label: 101, label: "chair" }`.
- `object_counts` has exactly the key `1`, and its value equals both the UI
  "Labelled object points" count and the number of coloured points in
  `scene_with_colored_objects.ply`.
- `mask_label_counts` still contains the raw model label `101` with the same
  count, so the part label is not lost.
- The background click has `obj_idx=0` and `is_positive=false`; the object
  click has `obj_idx=1` and `is_positive=true`.
- The PLY points are painted with the colour recorded in `objects[0].color`
  (8-bit tolerance 1), and every remaining point keeps the uniform 0.7 grey
  background.
- No `/recognition/mask_obj_recognition` request and no non-local host request
  is made while saving results.

The negative click uses unlabeled clutter inside the chair mask
(point 79693, `[2.90838, 1.54404, 0.18392]`), not the floor: a floor click is a
no-op for the real model because its mask never contains floor, so it cannot
regress the shrink behaviour. Both `maskBefore` and `maskAfter` are read from
the mask card, and the test asserts `after < before`.

## Important

- `MOCK_BACKEND=1` means a deterministic geometric rule produced the mask. It
  proves the **UI path**, not model inference. Do not cite its counts as
  PinPoint3D accuracy.
- The test places the camera through a DEV-only hook so the target chair is not
  occluded by furniture. Point picking itself still uses the app's normal
  raycasting and click handlers.
- The verified prompt points come from SNAP
  `data_examples/ScanNet/scene0025_00`:
  - positive: point index 74485, (3.121, 1.230, 0.897), instance 13 (chair)
  - negative: point index 79693, (2.908, 1.544, 0.184), unlabeled clutter
    inside the chair mask. The older floor point (index 46054) is still a valid
    input but is a near/no-op negative for the real model.
- Limitations: the backend keeps a single global session, so no other client
  may upload or infer while this test runs; the run needs the DEV-only
  projection hook (a Vite dev server, not a production build); the browser is
  the system Chrome (`PW_CHANNEL=chrome`); and the counts are raw model output
  on one scene, not an accuracy measurement.
