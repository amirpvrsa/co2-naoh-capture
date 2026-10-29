"""End-to-end: run.py builds the report, and the report keeps its honesty statements."""
import re
import subprocess
import sys

import yaml

from src.config import REPO_ROOT, load_config


def test_run_builds_report(tmp_path):
    cfg = load_config()
    cfg["outputs"]["dir"] = str(tmp_path / "out")
    cfg["sensitivity"] = {k: v[:2] for k, v in cfg["sensitivity"].items()}   # keep it quick
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, str(REPO_ROOT / "run.py"), "--config", str(path)],
                   check=True, cwd=REPO_ROOT)
    html = (tmp_path / "out" / cfg["report"]["file"]).read_text()
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", html, flags=re.S))
    assert "not</b> an Aspen Plus simulation" in html          # never passed off as Aspen
    assert "pure-water" in text                                  # solubility assumption stated
    assert "Model validity checks" in text
    assert (tmp_path / "out" / "stream_table.csv").exists()
