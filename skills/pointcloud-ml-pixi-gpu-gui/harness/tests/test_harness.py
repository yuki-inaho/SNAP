#!/usr/bin/env python3
"""Fixture checks for the synthetic point-cloud GUI Harness² task.

Run with the system interpreter; the planner and judge are stdlib-only and do not need the
Pixi/GPU environment. These checks validate the fixture's behavior, not a model's quality.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
BASE_PLANNER = HARNESS / "base" / "scripts" / "plan_gui_ml_test.py"
REFERENCE_PLANNER = HERE / "reference" / "plan_gui_ml_test.py"
JUDGE = HARNESS / "tools" / "judge.py"
EXECUTOR = HARNESS / "tools" / "answer.py"
TASKS = sorted(p for p in (HARNESS / "tasks").iterdir() if p.is_dir())
N_CRITERIA = 9


def run_planner(planner: Path, task: Path) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(planner), "--scenario", str(task / "source" / "scenario.json")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, json.loads(proc.stdout)


def run_judge(task: Path, answer_text: str) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        answer = Path(tmp) / "answer.json"
        answer.write_text(answer_text, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(JUDGE), "--task", str(task), "--answer", str(answer)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    return proc.returncode, json.loads(proc.stdout)


def verdict(result: dict) -> tuple[set[str], set[str]]:
    passed = {c["name"] for c in result["criteria"] if c["passed"]}
    failed = {c["name"] for c in result["criteria"] if not c["passed"]}
    return passed, failed


class FixtureTests(unittest.TestCase):
    def test_tasks_exist_and_are_synthetic(self) -> None:
        self.assertEqual(
            [p.name for p in TASKS], ["cpu-fallback", "sm75-multi", "sm90-single"]
        )
        for task in TASKS:
            scenario_text = (task / "source" / "scenario.json").read_text(encoding="utf-8")
            scenario = json.loads(scenario_text)
            self.assertTrue(scenario["scene_id"].startswith("synthetic-"))
            self.assertNotIn("/home/", scenario_text)
            self.assertNotIn("/tmp/", scenario_text)

    def test_expected_case_arithmetic(self) -> None:
        _, multi = run_planner(REFERENCE_PLANNER, HARNESS / "tasks" / "sm75-multi")
        self.assertEqual(multi["total_clicks"], 6)
        self.assertEqual(multi["memory_estimate_mb"], 784)
        self.assertEqual(multi["attention_backend"], "normal")
        self.assertEqual(len(multi["checks"]), 4)

        _, hopper = run_planner(REFERENCE_PLANNER, HARNESS / "tasks" / "sm90-single")
        self.assertEqual(hopper["attention_backend"], "flash")
        self.assertEqual(hopper["memory_estimate_mb"], 800)
        self.assertEqual(hopper["points_per_batch"], 4096)

        _, cpu = run_planner(REFERENCE_PLANNER, HARNESS / "tasks" / "cpu-fallback")
        self.assertEqual(cpu["points_per_batch"], 300)
        self.assertEqual(cpu["memory_estimate_mb"], 520)
        self.assertFalse(cpu["memory_ok"])
        self.assertEqual(len(cpu["checks"]), 2)

    def test_base_is_incomplete_on_every_scenario(self) -> None:
        for task in TASKS:
            with self.subTest(task=task.name):
                rc, plan = run_planner(BASE_PLANNER, task)
                self.assertEqual(rc, 0, "the base must run, only be incomplete")
                _, judged = run_judge(task, json.dumps(plan))
                passed, failed = verdict(judged)
                self.assertEqual(judged["n_criteria"], N_CRITERIA)
                self.assertFalse(judged["all_pass"])
                self.assertTrue(
                    {"memory", "checks", "gui_model_mode"} <= failed,
                    f"{task.name} unexpectedly kept base criteria: {sorted(passed)}",
                )

    def test_base_failure_shape_is_case_specific(self) -> None:
        _, base = run_planner(BASE_PLANNER, HARNESS / "tasks" / "sm90-single")
        _, judged = run_judge(HARNESS / "tasks" / "sm90-single", json.dumps(base))
        _, failed = verdict(judged)
        self.assertEqual(failed, {"memory", "gui_model_mode", "checks"})

        _, base_cpu = run_planner(BASE_PLANNER, HARNESS / "tasks" / "cpu-fallback")
        _, judged_cpu = run_judge(
            HARNESS / "tasks" / "cpu-fallback", json.dumps(base_cpu)
        )
        passed_cpu, failed_cpu = verdict(judged_cpu)
        self.assertTrue({"attention_backend", "clicks", "batching", "memory"} <= failed_cpu)
        self.assertEqual(passed_cpu, {"scene_id", "device", "seed"})

    def test_reference_satisfies_independent_judge(self) -> None:
        for task in TASKS:
            with self.subTest(task=task.name):
                rc, plan = run_planner(REFERENCE_PLANNER, task)
                self.assertEqual(rc, 0)
                rc, judged = run_judge(task, json.dumps(plan))
                self.assertEqual(rc, 0)
                self.assertEqual(judged["n_criteria"], N_CRITERIA)
                self.assertTrue(judged["all_pass"], judged)
                self.assertEqual(judged["n_passed"], N_CRITERIA)

    def test_executor_writes_an_answer_and_fails_loudly(self) -> None:
        task = HARNESS / "tasks" / "sm75-multi"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answer.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(EXECUTOR),
                    "--task",
                    str(task),
                    "--harness",
                    str(HARNESS / "base"),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.is_file())
            self.assertIsInstance(json.loads(out.read_text(encoding="utf-8")), dict)

            missing = Path(tmp) / "missing-planner"
            missing.mkdir()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(EXECUTOR),
                    "--task",
                    str(task),
                    "--harness",
                    str(missing),
                    "--out",
                    str(Path(tmp) / "missing.json"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("no usable planner JSON", proc.stderr)

    def test_malformed_answer_is_graded_not_crashed(self) -> None:
        rc, judged = run_judge(HARNESS / "tasks" / "sm75-multi", "{not json")
        self.assertEqual(rc, 0)
        self.assertEqual(judged["n_criteria"], N_CRITERIA)
        self.assertEqual(judged["n_passed"], 0)
        self.assertFalse(judged["all_pass"])

    def test_judge_is_separated_from_harness_and_answers(self) -> None:
        source = JUDGE.read_text(encoding="utf-8")
        self.assertNotIn("harness_dir", source)
        self.assertNotIn("--harness", source)
        self.assertNotIn("base/scripts", source)

        manifest = json.loads((HARNESS / "manifest.json").read_text(encoding="utf-8"))
        for pattern in ("expected*", "*.expected.json", "judge.json", "reference*"):
            self.assertIn(pattern, manifest["leak_globs"])

        for task in TASKS:
            for path in task.rglob("*"):
                self.assertNotIn("expected", path.name.lower())
                self.assertNotIn("reference", path.name.lower())
        for path in (HARNESS / "base").rglob("*"):
            self.assertNotIn("reference", path.name.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
