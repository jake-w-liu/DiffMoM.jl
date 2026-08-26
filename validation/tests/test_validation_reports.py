from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
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

    def meep_identity(self, prefix: str) -> dict[str, object]:
        return {
            "output_prefix": prefix,
            "periodic_bc_model": "bloch",
            "frequency_ghz": 10.0,
            "lambda_m": 0.03,
            "dx_cell_m": 0.036,
            "dy_cell_m": 0.036,
            "dx_lambda": 1.2,
            "dy_lambda": 1.2,
            "nx": 1,
            "ny": 1,
            "slot_wx_frac": 0.2,
            "slot_wy_frac": 0.2,
            "metal_fill_fraction": 1.0,
        }

    def write_meep_artifact_chain(
        self,
        data: Path,
        prefix: str,
        *,
        identity_overrides: dict[str, object] | None = None,
        reference_metrics: dict[str, object] | None = None,
        result_metrics: dict[str, object] | None = None,
    ) -> tuple[Path, Path, Path]:
        identity = self.meep_identity(prefix)
        identity.update(identity_overrides or {})
        nx = int(identity["nx"])
        ny = int(identity["ny"])
        geometry_path = data / f"julia_{prefix}_geometry.json"
        geometry_path.write_text(
            json.dumps(
                {
                    **identity,
                    "metal_mask_row_major": [[1] * nx for _ in range(ny)],
                },
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        reference_path = data / f"julia_{prefix}_reference.json"
        reference_path.write_text(
            json.dumps(
                {**identity, **(reference_metrics or {})},
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        result_path = data / f"meep_{prefix}_results.json"
        result_path.write_text(
            json.dumps(
                {
                    "output_prefix": prefix,
                    "julia_geometry_sha256": hashlib.sha256(
                        geometry_path.read_bytes()
                    ).hexdigest(),
                    "julia_reference_sha256": hashlib.sha256(
                        reference_path.read_bytes()
                    ).hexdigest(),
                    **(result_metrics or {}),
                },
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return geometry_path, reference_path, result_path

    def run_script(
        self, script: Path, project_root: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return self.run_cli(script, "--project-root", str(project_root), *args)

    def run_cli(
        self, script: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
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

    def test_csv_reader_rejects_ambiguous_record_structure(self) -> None:
        common = load_module(
            "validation_bempp_common_csv",
            BEMPP_DIR / "_bempp_common.py",
        )
        cases = {
            "duplicate column name": (
                "theta_deg,theta_deg,phi_deg,value\n0,90,5,1\n",
                "duplicate column name",
            ),
            "empty column name": (
                "theta_deg,phi_deg,value,\n0,5,1,unused\n",
                "empty column name",
            ),
            "extra record field": (
                "theta_deg,phi_deg,value\n0,5,1,unexpected\n",
                "beyond the header",
            ),
            "missing record field": (
                "theta_deg,phi_deg,value\n0,5\n",
                "missing value",
            ),
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, (contents, expected) in cases.items():
                with self.subTest(case=name):
                    path = root / f"{name.replace(' ', '_')}.csv"
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, expected):
                        common.load_csv_rows(path)

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
        plotter = load_module(
            "validation_impedance_plot",
            BEMPP_DIR / "plot_impedance_comparison.py",
        )
        theta = np.array([0.0, 0.0, 30.0, 30.0])
        phi = np.array([-5.0, 5.0, -5.0, 5.0])
        julia = np.array([1.0, 3.0, 5.0, 9.0])
        bempp = np.array([2.0, 4.0, 8.0, 10.0])

        cut_theta, cut_julia, cut_bempp = plotter.nearest_phi_cut(
            theta, phi, julia, bempp
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
            )

    def test_validation_python_help_does_not_require_optional_solvers(self) -> None:
        scripts = sorted(BEMPP_DIR.glob("*.py")) + sorted(MEEP_DIR.glob("*.py"))
        scripts = [path for path in scripts if not path.name.startswith("_")]
        for script in scripts:
            with self.subTest(script=script.name):
                result = self.run_cli(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--project-root", result.stdout)

    def test_invalid_numeric_options_fail_during_argument_parsing(self) -> None:
        cases = [
            (BEMPP_DIR / "run_pec_cross_validation.py", ["--freq-ghz", "nan"]),
            (
                BEMPP_DIR / "run_impedance_cross_validation.py",
                ["--n-phi", "0"],
            ),
            (
                BEMPP_DIR / "sweep_impedance_conventions.py",
                ["--mesh-step-lambda", "-1"],
            ),
            (
                MEEP_DIR / "run_periodic_cross_validation.py",
                ["--resolution", "0"],
            ),
            (
                MEEP_DIR / "run_reflectance_curve_comparison.py",
                ["--slot-wx-fracs", "1.1"],
            ),
        ]
        for script, arguments in cases:
            with self.subTest(script=script.name):
                result = self.run_cli(script, *arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

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
            self.assertIn("standard JSON", result.stderr)

    def test_operator_report_rejects_overflowed_json_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            self.write_current_pair(data, "case")
            (data / "julia_case_operator_checks.json").write_text(
                '{"solve_residual_l2_rel": 1e999}\n', encoding="utf-8"
            )

            result = self.run_script(
                BEMPP_DIR / "compare_impedance_operator_aligned.py",
                root,
                "--output-prefix",
                "case",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-finite number", result.stderr)

    def test_json_readers_reject_duplicate_object_members(self) -> None:
        bempp_common = load_module(
            "validation_bempp_common_json",
            BEMPP_DIR / "_bempp_common.py",
        )
        meep_common = load_module(
            "validation_meep_common_json",
            MEEP_DIR / "_meep_common.py",
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(
                '{"metrics": {"residual": 0.1, "residual": 0.9}}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                bempp_common.ArtifactReadError,
                "duplicate object member 'residual'",
            ):
                bempp_common.read_json_object(path)
            with self.assertRaisesRegex(
                SystemExit,
                "duplicate object member 'residual'",
            ):
                meep_common.load_json_object(
                    path,
                    recovery="Regenerate the source artifact.",
                )

    def test_meep_comparison_fails_only_the_primary_reflectance_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()

            def write_case(prefix: str, julia_refl: float, meep_refl: float) -> None:
                self.write_meep_artifact_chain(
                    data,
                    prefix,
                    reference_metrics={
                        "refl_total_fraction": julia_refl,
                        "trans_total_fraction_closure": 1.0 - julia_refl,
                        "abs_total_fraction": 0.0,
                    },
                    result_metrics={
                        "reflectance_total": meep_refl,
                        "transmittance_total": 0.0,
                        "absorption_total": 0.0,
                    },
                )

            write_case("reflectance_fail", julia_refl=0.0, meep_refl=1.0)
            failed = self.run_script(
                MEEP_DIR / "compare_periodic_to_julia.py",
                root,
                "--output-prefix",
                "reflectance_fail",
                "--tol-refl",
                "0.1",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Reflectance comparison failed", failed.stderr)
            failed_report = self.load_standard_json(
                data / "meep_reflectance_fail_cross_validation_report.json"
            )
            self.assertEqual(failed_report["verdict"], "CHECK")

            write_case("trans_diagnostic", julia_refl=0.5, meep_refl=0.5)
            passed = self.run_script(
                MEEP_DIR / "compare_periodic_to_julia.py",
                root,
                "--output-prefix",
                "trans_diagnostic",
                "--tol-trans",
                "0.1",
            )
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            passed_report = self.load_standard_json(
                data / "meep_trans_diagnostic_cross_validation_report.json"
            )
            self.assertEqual(passed_report["verdict"], "PASS")
            self.assertEqual(passed_report["trans_verdict"], "CHECK")

            reference_path = data / "julia_trans_diagnostic_reference.json"
            changed_reference = json.loads(reference_path.read_text(encoding="utf-8"))
            changed_reference["abs_total_fraction"] = 0.25
            reference_path.write_text(
                json.dumps(changed_reference, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            stale = self.run_script(
                MEEP_DIR / "compare_periodic_to_julia.py",
                root,
                "--output-prefix",
                "trans_diagnostic",
            )
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn("does not match current artifact", stale.stderr)

    def test_meep_runner_rejects_mismatched_julia_artifacts_before_solver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            prefix = "identity_mismatch"
            identity = self.meep_identity(prefix)
            (data / f"julia_{prefix}_geometry.json").write_text(
                json.dumps(
                    {**identity, "metal_mask_row_major": [[1]]},
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (data / f"julia_{prefix}_reference.json").write_text(
                json.dumps({**identity, "nx": 2}, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            result = self.run_script(
                MEEP_DIR / "run_periodic_cross_validation.py",
                root,
                "--output-prefix",
                prefix,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("artifact identity mismatch", result.stderr)
            self.assertNotIn("Could not import Meep", result.stderr)

    def test_single_point_meep_correlation_is_unavailable(self) -> None:
        analyzer = load_module(
            "meep_detailed_analysis",
            MEEP_DIR / "analyze_meep_detailed_comparison.py",
        )
        metrics = analyzer.compute_metrics([{"julia_refl": 0.5, "meep_refl": 0.4}])
        self.assertIsNone(metrics["corr"])
        json.dumps(metrics, allow_nan=False)

    def test_curve_runner_reports_invalid_width_list_without_traceback(self) -> None:
        result = self.run_cli(
            MEEP_DIR / "run_reflectance_curve_comparison.py",
            "--slot-wx-fracs",
            "not-a-number",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--slot-wx-fracs", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_curve_reuse_rechecks_verdict_and_accepts_closure_only(self) -> None:
        curve = load_module(
            "meep_curve_reuse_validation",
            MEEP_DIR / "run_reflectance_curve_comparison.py",
        )
        curve.make_plot = (
            lambda rows, output, tol_refl: output.write_bytes(b"plot")
        )

        for verdict in ("CHECK", "PASS"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / "data"
                data.mkdir()
                case_prefix = "curve_wx0p200"
                self.write_meep_artifact_chain(
                    data,
                    case_prefix,
                    identity_overrides={
                        "nx": 14,
                        "ny": 14,
                        "slot_wx_frac": 0.2,
                    },
                    reference_metrics={
                        "refl_total_fraction": 0.1,
                        "trans_total_fraction_closure": 0.9,
                    },
                    result_metrics={
                        "resolution_px_per_lambda": 30,
                        "pml_lambda": 1.0,
                        "sz_lambda": 6.0,
                        "requested_metal_thickness_lambda": 0.03,
                        "source_offset_lambda": 0.35,
                        "refl_offset_lambda": 0.25,
                        "tran_offset_lambda": 0.35,
                        "fwidth": 0.2,
                        "after_sources_time": 180.0,
                        "reflectance_total": 0.1,
                        "transmittance_total": 0.9,
                    },
                )
                report_path = (
                    data
                    / f"meep_{case_prefix}_cross_validation_report.json"
                )
                report_path.write_text(
                    json.dumps(
                        {
                            "abs_diff_refl": 0.0,
                            "abs_diff_trans": 0.0,
                            "verdict": verdict,
                        },
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                commands: list[list[str]] = []
                curve.run_command = lambda command, cwd: commands.append(command)
                arguments = [
                    str(MEEP_DIR / "run_reflectance_curve_comparison.py"),
                    "--project-root",
                    str(root),
                    "--prefix-base",
                    "curve",
                    "--slot-wx-fracs",
                    "0.2",
                    "--reuse-existing",
                ]
                with mock.patch.object(sys, "argv", arguments):
                    if verdict == "CHECK":
                        with self.assertRaisesRegex(SystemExit, "is not PASS"):
                            curve.main()
                    else:
                        curve.main()

                self.assertEqual(len(commands), 1)
                self.assertIn("compare_periodic_to_julia.py", commands[0][1])
                summary_path = data / "curve_curve_summary.json"
                if verdict == "CHECK":
                    self.assertFalse(summary_path.exists())
                else:
                    summary = self.load_standard_json(summary_path)
                    self.assertEqual(summary["rows"][0]["julia_trans_total"], 0.9)

                mismatched_arguments = [*arguments, "--nx", "15"]
                with mock.patch.object(sys, "argv", mismatched_arguments):
                    with self.assertRaisesRegex(
                        SystemExit, "does not match current --nx"
                    ):
                        curve.main()

    def test_meep_subprocess_failure_has_actionable_exit(self) -> None:
        common = load_module(
            "meep_common_subprocess_failure",
            MEEP_DIR / "_meep_common.py",
        )
        failure = subprocess.CalledProcessError(7, ["solver", "--case", "bad"])
        with mock.patch.object(common.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(SystemExit, "exited with status 7"):
                common.run_command(["solver", "--case", "bad"], REPO_ROOT)

    def test_meep_analyzer_rejects_empty_case_lists_without_traceback(self) -> None:
        result = self.run_cli(
            MEEP_DIR / "analyze_meep_detailed_comparison.py",
            "--curve-suffixes",
            "",
            "--conv-prefixes",
            "",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--curve-suffixes", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
