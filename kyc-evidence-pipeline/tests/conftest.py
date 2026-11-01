from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(tmp_path):
    return {"db": str(tmp_path / "evidence.sqlite"), "cp": str(tmp_path / "checkpoints.sqlite"), "tmp": tmp_path}


@pytest.fixture
def cli():
    def run(*arguments: str, expected: int = 0):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-m", "kyc_pipeline", *map(str, arguments)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        assert proc.returncode == expected, f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        return json.loads(proc.stdout) if expected == 0 else json.loads(proc.stderr)
    return run


def defer_all(result: dict, tmp: Path) -> Path:
    d = {"reviewer_id": "TEST_ANALYST", "decisions": [{"issue_id": i["issue_id"], "action": "defer", "reason": "Fictional test case needs further evidence."} for i in result["review_request"]["issues"]]}
    path = tmp / "decision.json"
    path.write_text(json.dumps(d))
    return path
