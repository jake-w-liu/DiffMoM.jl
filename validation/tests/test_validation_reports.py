from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BEMPP_DIR = REPO_ROOT / "validation" / "bempp"
MEEP_DIR = REPO_ROOT / "validation" / "meep"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reject_json_constant(value: str):
    raise ValueError(f"non-standard numeric constant {value}")


class ValidationReportTests(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_script(
        self, script: Path, project_root: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(script),
                "--project-root",
                str(project_root),
                *args,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def load_standard_json(self, path: Path):
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_json_constant,
        )

    def test_pec_uses_nearest_sampled_phi_cut_and_standard_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            julia_rows = [
                {"theta_deg": 5, "phi_deg": 5, "dir_pec_dBi": 1.0},
                {"theta_deg": 30, "phi_deg": 5, "dir_pec_dBi": 2.0},
                {"theta_deg": 5, "phi_deg": 20, "dir_pec_dBi": 3.0},
            ]
            bempp_rows = [
                {"theta_deg": 5, "phi_deg": 5, "dir_bempp_dBi": 1.5},
                {"theta_deg": 30, "phi_deg": 5, "dir_bempp_dBi": 1.5},
                {"theta_deg": 5, "phi_deg": 20, "dir_bempp_dBi": 13.0},
            ]
            self.write_csv(
                data / "beam_steer_farfield.csv",
                ["theta_deg", "phi_deg", "dir_pec_dBi"],
                julia_rows,
            )
            self.write_csv(
                data / "bempp_pec_farfield.csv",
                ["theta_deg", "phi_deg", "dir_bempp_dBi"],
                bempp_rows,
            )

            result = self.run_script(
                BEMPP_DIR / "compare_pec_to_julia.py",
                root,
                "--target-theta-deg",
                "45",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = self.load_standard_json(
                data / "bempp_cross_validation_report.json"
            )
            self.assertEqual(report["phi0_cut"]["nearest_phi_abs_deg"], 5.0)
            self.assertEqual(report["phi0_cut"]["num_points"], 2)
            self.assertEqual(report["phi0_cut"]["mean_abs_diff_db"], 0.5)
            markdown = (data / "bempp_cross_validation_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Near 45 deg", markdown)

    def test_pec_rejects_nonfinite_csv_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            self.write_csv(
                data / "beam_steer_farfield.csv",
                ["theta_deg", "phi_deg", "dir_pec_dBi"],
                [{"theta_deg": 5, "phi_deg": 5, "dir_pec_dBi": "nan"}],
            )
            self.write_csv(
                data / "bempp_pec_farfield.csv",
                ["theta_deg", "phi_deg", "dir_bempp_dBi"],
                [{"theta_deg": 5, "phi_deg": 5, "dir_bempp_dBi": 1.0}],
            )

            result = self.run_script(BEMPP_DIR / "compare_pec_to_julia.py", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Non-finite numeric value", result.stderr)
            self.assertFalse((data / "bempp_cross_validation_report.json").exists())

    def test_impedance_uses_json_null_when_no_sidelobe_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            julia_rows = [
                {
                    "theta_deg": theta,
                    "phi_deg": 5,
                    "dir_julia_imp_dBi": level,
                }
                for theta, level in ((0, 1.0), (30, 2.0), (60, 3.0))
            ]
            bempp_rows = [
                {
                    "theta_deg": theta,
                    "phi_deg": 5,
                    "dir_bempp_imp_dBi": level + 0.25,
                }
                for theta, level in ((0, 1.0), (30, 2.0), (60, 3.0))
            ]
            self.write_csv(
                data / "julia_case_farfield.csv",
                ["theta_deg", "phi_deg", "dir_julia_imp_dBi"],
                julia_rows,
            )
            self.write_csv(
                data / "bempp_case_farfield.csv",
                ["theta_deg", "phi_deg", "dir_bempp_imp_dBi"],
                bempp_rows,
            )

            result = self.run_script(
                BEMPP_DIR / "compare_impedance_to_julia.py",
                root,
                "--output-prefix",
                "case",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = self.load_standard_json(
                data / "bempp_case_cross_validation_report.json"
            )
            features = report["pattern_features"]
            self.assertIsNone(features["julia_sll_down_db"])
            self.assertIsNone(features["bempp_sll_down_db"])
            self.assertIsNone(features["sll_down_diff_db"])
            markdown = (data / "bempp_case_cross_validation_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("not available", markdown)

    def test_plot_cut_averages_equidistant_azimuth_samples(self) -> None:
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.__path__ = []

        def use_backend(_backend: str) -> None:
            return None

        matplotlib.use = use_backend
        pyplot = types.ModuleType("matplotlib.pyplot")
        with mock.patch.dict(
            sys.modules,
            {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot},
        ):
            plotter = load_module(
                "validation_impedance_plot",
                BEMPP_DIR / "plot_impedance_comparison.py",
            )
        theta = np.array([0.0, 0.0, 30.0, 30.0])
        phi = np.array([-5.0, 5.0, -5.0, 5.0])
        julia = np.array([1.0, 3.0, 5.0, 9.0])
        bempp = np.array([2.0, 4.0, 8.0, 10.0])

        cut_theta, cut_julia, cut_bempp = plotter.nearest_phi_cut(
            theta, phi, julia, bempp, n_phi=2
        )

        np.testing.assert_array_equal(cut_theta, np.array([0.0, 30.0]))
        np.testing.assert_allclose(cut_julia, np.array([2.0, 7.0]))
        np.testing.assert_allclose(cut_bempp, np.array([3.0, 9.0]))

        with self.assertRaisesRegex(ValueError, "finite values"):
            plotter.nearest_phi_cut(
                theta,
                np.array([-5.0, 5.0, -5.0, np.nan]),
                julia,
                bempp,
                n_phi=2,
            )

    def test_matrix_rejects_report_without_sidelobe_metric(self) -> None:
        matrix_module = load_module(
            "validation_matrix",
            BEMPP_DIR / "run_impedance_validation_matrix.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            report = {
                "pattern_features": {
                    "main_theta_abs_diff_deg": 0.1,
                    "main_level_diff_db": 0.2,
                    "sll_down_diff_db": None,
                }
            }
            for case in matrix_module.CASES:
                path = data / f"bempp_{case.case_id}_cross_validation_report.json"
                path.write_text(
                    json.dumps(report, allow_nan=False) + "\n", encoding="utf-8"
                )

            result = self.run_script(
                BEMPP_DIR / "run_impedance_validation_matrix.py",
                root,
                "--skip-julia",
                "--skip-bempp",
                "--skip-compare",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid comparison report", result.stderr)
            self.assertIn("sll_down_diff_db", result.stderr)

    def write_current_pair(self, data: Path, prefix: str) -> None:
        fields = [
            "x_m",
            "y_m",
            "z_m",
            "Jx_re",
            "Jx_im",
            "Jy_re",
            "Jy_im",
            "Jz_re",
            "Jz_im",
        ]
        rows = [
            {
                "x_m": index * 0.001,
                "y_m": 0.0,
                "z_m": 0.0,
                "Jx_re": 1.0,
                "Jx_im": 0.0,
                "Jy_re": 0.0,
                "Jy_im": 0.0,
                "Jz_re": 0.0,
                "Jz_im": 0.0,
            }
            for index in range(8)
        ]
        self.write_csv(data / f"julia_{prefix}_element_currents.csv", fields, rows)
        self.write_csv(data / f"bempp_{prefix}_element_currents.csv", fields, rows)

    def test_operator_report_uses_null_for_absent_residual_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            self.write_current_pair(data, "case")

            result = self.run_script(
                BEMPP_DIR / "compare_impedance_operator_aligned.py",
                root,
                "--output-prefix",
                "case",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = self.load_standard_json(
                data / "bempp_case_operator_aligned_report.json"
            )
            self.assertIsNone(report["julia_residual_rel_l2"])
            self.assertIsNone(report["bempp_residual_rel_l2"])
            markdown = (data / "bempp_case_operator_aligned_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("not available", markdown)

    def test_operator_report_rejects_nonstandard_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            self.write_current_pair(data, "case")
            (data / "julia_case_operator_checks.json").write_text(
                '{"solve_residual_l2_rel": NaN}\n', encoding="utf-8"
            )

            result = self.run_script(
                BEMPP_DIR / "compare_impedance_operator_aligned.py",
                root,
                "--output-prefix",
                "case",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("standard JSON metadata", result.stderr)

    def test_single_point_meep_correlation_is_unavailable(self) -> None:
        analyzer = load_module(
            "meep_detailed_analysis",
            MEEP_DIR / "analyze_meep_detailed_comparison.py",
        )
        metrics = analyzer.compute_metrics([{"julia_refl": 0.5, "meep_refl": 0.4}])
        self.assertIsNone(metrics["corr"])
        json.dumps(metrics, allow_nan=False)

    def test_curve_runner_reports_invalid_width_list_without_traceback(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(MEEP_DIR / "run_reflectance_curve_comparison.py"),
                "--slot-wx-fracs",
                "not-a-number",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--slot-wx-fracs", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_meep_analyzer_rejects_empty_case_lists_without_traceback(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(MEEP_DIR / "analyze_meep_detailed_comparison.py"),
                "--curve-suffixes",
                "",
                "--conv-prefixes",
                "",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--curve-suffixes", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
