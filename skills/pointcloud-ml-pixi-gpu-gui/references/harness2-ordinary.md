# Bundled Harness² ordinary-task fixture

`harness/` is a synthetic, reusable ordinary-task fixture for the GPU GUI test-planning
workflow. It is not benchmark accuracy evidence and it is not a sandbox. It plans a test
matrix; it does not run the GUI or a real checkpoint. The planned `gui_model_mode: "fake"`
describes the fast interaction layer only, not a claim that real GUI operation is
unnecessary. The offscreen command in the plan is a default; on hosts where Qt/VTK needs an
X display, replace it with an Xvfb-backed command and record that change.

## What is fixed

The manifest pins the comparison conditions:

- editor channel: installed OpenCode CLI, `service` launcher;
- editor model: `ollama-cloud/deepseek-v4.1-flash:1017#high`;
- `k=1`, `--modes parallel`;
- executor and judge timeouts, executor argv, task inputs and base harness;
- the judge is a separate command that receives only the task prompt/source and a copy of
  the answer, never the harness or the evidence tree.

Changing the model, budget, base harness or inputs invalidates a same-condition
comparison. Do not switch them silently mid-run.

## Layout

```text
harness/
|-- manifest.json
|-- base/                 editable harness (systemprompt + planner script)
|-- tasks/                one synthetic scenario per task
|-- tools/                executor and independent judge
`-- tests/                fixture checks and a reference planner
```

The reference planner lives under `tests/reference/` and is not part of `base/`, so it is
never copied into an editor workspace. Expected values are recomputed by `tools/judge.py`
from the task source; there is no expected-answer file in the task tree.

The executor is a fixed wrapper, not an agent. It runs whatever planner is in the harness
tree. Editing `systemprompt.md` or adding a skill changes the advice available to an
agent, but it cannot change this executor's output; the planner code is the lever.

The mission asks for the **plan**, not for the planner source. The planner is the harness
component the executor invokes. If the produced plan violates the contract, the harness
editor should change the planner code. A framing that asks the editor to "repair the
planner" can be read as the solver's deliverable and produce `KEEP_BASE`; the fixture keeps
the deliverable request and the harness lever separate for that reason.

## Commands

Use the installed `harness2-ordinary` binary from the Harness² fork environment. A dry run
writes nothing and prints the resolved plan:

```bash
harness2-ordinary \
  --manifest skills/pointcloud-ml-pixi-gpu-gui/harness/manifest.json \
  --output <fresh-directory-outside-repo> \
  --k 1 --modes parallel --dry-run
```

A real run needs a fresh output directory (no resume):

```bash
harness2-ordinary \
  --manifest skills/pointcloud-ml-pixi-gpu-gui/harness/manifest.json \
  --output <fresh-directory-outside-repo> \
  --k 1 --modes parallel
```

For a smaller probe, add `--only-tasks <id>` (comma-separated). That changes the task set;
record it as a different condition rather than comparing it to the full run.

Exit `0` means every task finished, `1` means at least one task did not (including an
editor task whose calls all failed), and `2` means misconfiguration. Read
`report.json`, `editor_health`, per-task `integration`, and the receipts under
`audit/sessions/`. A green native status with `all_failed` editor health is not editor
success.

## Fixture self-checks

Run the fixture checks with the system `python3`; they do not need the GPU environment:

```bash
python3 skills/pointcloud-ml-pixi-gpu-gui/harness/tests/test_harness.py
```

The checks verify that the base planner is deliberately incomplete on every scenario, the
reference planner satisfies the independent judge, malformed answers are graded as failed
criteria rather than a crash, and the judge does not depend on the harness tree.

## Honest limits

- `k=1` selection is candidate selection, not measured improvement. A base/adopted delta
  is reported only when both runs are judged under matching configured conditions; it is
  still not a real-world generalization claim.
- The editor's tool-permission fence is not an OS sandbox. Fresh sessions and separate
  workspaces reduce interference; they do not prove isolation.
- Raw editor events and model output can contain private task data. Review them before
  publishing anything.
- An unmeasured editor cost is reported as `null`, never zero.
- The configured model id is what was launched. Reasoning effort/variant visibility depends
  on the installed CLI and its event stream; do not claim to inspect hidden reasoning.
- Validation history: an early framing that asked the editor to "repair the planner"
  produced `KEEP_BASE` and a judged delta of 0 even though every editor session succeeded.
  After the mission was reframed as a deliverable request, one synthetic task measured
  4/9 to 9/9 (delta 5/9) as a same-condition judged pair. That is one case, not a
  generalization; other scenarios were covered by fixture tests and by a separate all-task
  run under the earlier framing. Re-run after any prompt or base change and record it as a
  new condition.
