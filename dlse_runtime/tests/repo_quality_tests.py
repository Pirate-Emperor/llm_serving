#!/usr/bin/env python3
"""Check repository layout, README links, and figure readability."""

from __future__ import annotations

import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
SVG_NS = "{http://www.w3.org/2000/svg}"


class RepositoryQualityTests(unittest.TestCase):
    def test_one_root_license(self) -> None:
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertFalse((ROOT / "LICENSE.txt").exists())
        self.assertFalse((ROOT / "LICENSE_CUDA").exists())
        self.assertIn("Pirate-Emperor", (ROOT / "LICENSE").read_text())

    def test_removed_overview_image(self) -> None:
        self.assertFalse((ROOT / "main.png").exists())
        self.assertNotIn("main.png", README.read_text())

    def test_clone_instructions_use_default_branch(self) -> None:
        text = README.read_text()
        self.assertIn("git clone https://github.com/SpryzenHell/llm_serving.git", text)
        self.assertNotIn("git checkout feat/dlse-runtime-revamp", text)

    def test_readme_local_links_exist(self) -> None:
        text = re.sub(r"~~~.*?~~~", "", README.read_text(), flags=re.DOTALL)
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
        for target in links:
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            self.assertTrue((ROOT / path).exists(), f"broken README link: {target}")

    def test_figures_parse_and_have_readable_labels(self) -> None:
        assets = ROOT / "docs" / "assets"
        expected = {
            "architecture.svg", "disaggregated_flow.svg", "host_validation.svg",
            "scheduler_trace.svg", "kv_reservation.svg", "test_matrix.svg", "scheduler_iterations.svg",
            "scheduler_sweep.svg", "kv_sweep.svg", "execution_snapshot.svg",
        }
        self.assertTrue(expected.issubset({p.name for p in assets.glob("*.svg")}))
        for path in assets.glob("*.svg"):
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, SVG_NS + "svg", path.name)
            texts = root.findall(".//" + SVG_NS + "text")
            self.assertTrue(texts, f"figure has no labels: {path.name}")
            def check_inherited_sizes(node: ET.Element, inherited: float | None = None) -> None:
                declared = node.attrib.get("font-size")
                size = inherited
                if declared is not None:
                    size = float(re.sub(r"[^0-9.]", "", declared))
                if node.tag == SVG_NS + "text":
                    self.assertIsNotNone(size, f"missing font size on text or parent: {path.name}")
                    self.assertGreaterEqual(float(size), 16.0, f"small SVG font: {path.name}")
                for child in node:
                    check_inherited_sizes(child, size)

            check_inherited_sizes(root)
            for node in texts:
                fill = node.attrib.get("fill")
                self.assertIsNotNone(fill, f"missing text fill: {path.name}")
                self.assertRegex(
                    fill or "",
                    r"^(#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}|black|white|currentColor)$",
                    f"invalid or invisible text fill in {path.name}: {fill!r}",
                )
            if path.name == "scheduler_iterations.svg":
                sizes = [
                    float(re.sub(r"[^0-9.]", "", node.attrib["font-size"]))
                    for node in texts if "font-size" in node.attrib
                ]
                self.assertGreaterEqual(min(sizes), 18.0, "scheduler iteration labels should be legible")

    def test_terminal_figures_use_shell_style_without_window_dots(self) -> None:
        for name in ("host_validation.svg", "execution_snapshot.svg"):
            source = (ROOT / "docs" / "assets" / name).read_text()
            self.assertIn("runner@ubuntu:~/llm_serving", source)
            self.assertIn("DejaVu Sans Mono", source)
            self.assertNotIn("<circle", source, f"{name} should not mimic macOS window controls")

    def test_required_runtime_files_exist(self) -> None:
        required = [
            "dlse_runtime/src/runtime.cpp",
            "dlse_runtime/tests/runtime_tests.cpp",
            "dlse_runtime/tests/repo_quality_tests.py",
            "dlse_runtime/benchmarks/runtime_experiments.cpp",
            "dlse_runtime/benchmarks/runtime_bench.cpp",
            "dlse_runtime/cuda/paged_attention.cu",
            ".github/workflows/dlse-host.yml",
        ]
        for path in required:
            self.assertTrue((ROOT / path).is_file(), f"missing required file: {path}")

    def test_helper_scripts_are_executable(self) -> None:
        for path in (ROOT / "dlse_runtime" / "scripts").glob("*.sh"):
            self.assertTrue(path.stat().st_mode & 0o111, f"not executable: {path.name}")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RepositoryQualityTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
