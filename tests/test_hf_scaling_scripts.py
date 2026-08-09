import csv
import hashlib
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = REPO_ROOT / "runs" / "prepare_data_hf.sh"
SCALING_SCRIPT = REPO_ROOT / "runs" / "scaling_laws_hf.sh"
ORIGINAL_SCALING_SCRIPT = REPO_ROOT / "runs" / "scaling_laws.sh"
TEST_HF_ENV = {
    "HF_NAMESPACE": "pytest-user",
    "HF_BUCKET": "pytest-user/nanochat-scaling-laws",
    "GIT_REPO_URL": "https://github.com/example/nanochat.git",
}


def run_script(script, *args, env=None, check=True):
    process_env = os.environ.copy()
    process_env.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        check=check,
    )


@pytest.fixture
def training_log(tmp_path):
    path = tmp_path / "training-output.txt"
    path.write_text(
        """\
wte                     : 25,165,824
value_embeds            : 12,582,912
lm_head                 : 25,165,824
transformer_matrices    : 80,000,000
scalars                 : 30
total                   : 142,914,590
Calculated number of iterations from target FLOPs: 2
Total training FLOPs estimate: 1.234500e+15
Total batch size 524,288 => gradient accumulation steps: 1
Step 00002 | Validation bpb: 2.345678
Step 00002 | CORE metric: 0.1234
step 00000/00002 (0.00%) | loss: 10.0 | tok/sec: 123,456 | bf16_mfu: 34.56 | epoch: 0
"""
    )
    return path


def test_shell_syntax():
    for script in (PREP_SCRIPT, SCALING_SCRIPT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_original_scaling_script_is_untouched():
    digest = hashlib.sha256(ORIGINAL_SCALING_SCRIPT.read_bytes()).hexdigest()
    assert digest == "f510a904dca15271b9c1909630c047358500f92007c31018e2723f07c9cb48d3"


def test_production_requires_run_label():
    result = run_script(
        SCALING_SCRIPT,
        env={"RUN_LABEL": "", **TEST_HF_ENV},
        check=False,
    )
    assert result.returncode != 0
    assert "RUN_LABEL is required" in result.stderr

@pytest.mark.parametrize("script", [PREP_SCRIPT, SCALING_SCRIPT])
def test_job_id_parser_accepts_hf_cli_output(script):
    result = subprocess.run(
        ["bash", str(script), "--parse-job-id"],
        cwd=REPO_ROOT,
        input="✓ Job started\n  id: 6a78e49e3e1f34a7e32c100e\n  name: nanochat-scaling\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "6a78e49e3e1f34a7e32c100e"


@pytest.mark.parametrize("script", [PREP_SCRIPT, SCALING_SCRIPT])
def test_job_id_parser_rejects_unrecognized_output(script):
    result = subprocess.run(
        ["bash", str(script), "--parse-job-id"],
        cwd=REPO_ROOT,
        input="Job submission output without an identifier\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout == ""


def test_metric_parser_accepts_complete_log(training_log):
    result = run_script(
        SCALING_SCRIPT,
        "--parse-log",
        str(training_log),
        "1e18",
        "10",
        "scaling_pytest-parser_1e18_d10",
        "42",
    )
    row = next(csv.reader([result.stdout]))
    assert len(row) == 18
    assert row[0] == "scaling_pytest-parser_1e18_d10"
    assert row[1:5] == ["1e18", "1.2345e+15", "10", "640"]
    assert row[11:18] == ["2", "1048576", "2.345678", "0.1234", "123456", "34.56", "42"]


def test_metric_parser_rejects_missing_core_metric(training_log):
    training_log.write_text(training_log.read_text().replace("Step 00002 | CORE metric: 0.1234\n", ""))
    result = run_script(
        SCALING_SCRIPT,
        "--parse-log",
        str(training_log),
        "1e18",
        "10",
        "scaling_pytest-parser_1e18_d10",
        "42",
        check=False,
    )
    assert result.returncode != 0
    assert "Missing CORE score" in result.stderr
