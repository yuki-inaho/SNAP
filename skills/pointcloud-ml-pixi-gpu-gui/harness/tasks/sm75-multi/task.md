# Mission: produce a GPU GUI test plan

Produce one JSON test plan for the scenario in `source/scenario.json`. Run the planner with:

```bash
python3 scripts/plan_gui_ml_test.py --scenario source/scenario.json
```

and report the JSON it prints. Work from the scenario file, never from memory.

## Output contract

| Key | Value |
|---|---|
| `scene_id` | Copy `scene_id` from the scenario. |
| `device` | `"cuda"` when `gpu.available` is true, otherwise `"cpu"`. |
| `attention_backend` | `"normal"` on CPU. On CUDA, `"flash"` only when `attention_installed` is true **and** `gpu.compute_capability >= 8.0`; otherwise `"normal"`. |
| `seed` | Copy `seed`. |
| `click_rounds` | Number of entries in `prompt_rounds`. |
| `total_clicks` | Sum of every round's `clicks`. |
| `points_per_batch` | `min(points, batch_points)`. |
| `num_batches` | `ceil(points / points_per_batch)`. |
| `memory_estimate_mb` | `512 + 8 * ceil(points / 1024) + (256 if device is cuda else 0)`. |
| `memory_ok` | `memory_estimate_mb <= memory_budget_mb`. |
| `gui_model_mode` | Always `"fake"`: interaction tests use a deterministic fake model; real inference belongs in a separate smoke test. |
| `checks` | Exact ordered list described below. |

`checks` always begins with:

```json
[
  "pixi run just imports",
  "QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true pixi run pytest -q tests/test_gui_interaction.py"
]
```

When `device` is `"cuda"`, append, in this order:

```json
"pixi run just validate",
"pixi run just model-smoke"
```

Do not append GPU-only commands for a CPU plan. Do not add other commands.

## Constraints

- Use only the Python standard library.
- Print only the JSON object to stdout; send any diagnostic to stderr.
- Do not write files, access the network, or launch GPU work.
- Keep the planner deterministic: the same scenario must produce the same plan.
- Do not change the required key names or command strings.
