import json

import pytest
import torch

from nanochat import core_eval
from scripts import base_eval


def test_evaluate_task_limits_targets_but_keeps_fewshot_pool(monkeypatch):
    data = [{"index": index} for index in range(12)]
    calls = []

    def fake_evaluate_example(idx, model, tokenizer, example_pool, device, task_meta):
        calls.append((idx, len(example_pool)))
        return idx == 0

    monkeypatch.setattr(core_eval, "evaluate_example", fake_evaluate_example)

    accuracy = core_eval.evaluate_task(
        model=None,
        tokenizer=None,
        data=data,
        device=torch.device("cpu"),
        task_meta={},
        max_examples=2,
    )

    assert calls == [(0, 12), (1, 12)]
    assert accuracy == pytest.approx(0.5)


def test_evaluate_core_passes_limit_without_truncating_task_data(tmp_path, monkeypatch):
    eval_bundle = tmp_path / "eval_bundle"
    eval_data = eval_bundle / "eval_data"
    eval_data.mkdir(parents=True)
    (eval_bundle / "core.yaml").write_text(
        """icl_tasks:
  - label: test_task
    icl_task_type: multiple_choice
    dataset_uri: test.jsonl
    num_fewshot: [10]
""",
        encoding="utf-8",
    )
    (eval_bundle / "eval_meta_data.csv").write_text(
        "Eval Task,Random baseline\ntest_task,25\n",
        encoding="utf-8",
    )
    examples = [{"index": index} for index in range(12)]
    (eval_data / "test.jsonl").write_text(
        "".join(json.dumps(example) + "\n" for example in examples),
        encoding="utf-8",
    )
    observed = {}

    def fake_evaluate_task(model, tokenizer, data, device, task_meta, max_examples=-1):
        observed["pool_size"] = len(data)
        observed["max_examples"] = max_examples
        return 0.5

    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "evaluate_task", fake_evaluate_task)

    result = base_eval.evaluate_core(None, None, torch.device("cpu"), max_per_task=1)

    assert observed == {"pool_size": 12, "max_examples": 1}
    assert result["results"] == {"test_task": 0.5}
