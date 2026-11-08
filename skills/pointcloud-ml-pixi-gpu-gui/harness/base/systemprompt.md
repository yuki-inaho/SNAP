# GPU GUI test-plan harness

Produce the requested GPU GUI test plan with:

```bash
python3 scripts/plan_gui_ml_test.py --scenario source/scenario.json
```

Report the JSON it prints. Work from the scenario file, never from memory. The executor is
fixed and copies the planner's stdout as the answer, so a plan that violates the contract is
a harness defect: change the planner code in `scripts/`, not only the surrounding prose.

Keep the planner deterministic and dependency-free (stdlib only). It may read the scenario
and print one JSON object to stdout; it must not write files, access the network, or launch
GPU work. Preserve the required key names and the exact command strings in the contract.
