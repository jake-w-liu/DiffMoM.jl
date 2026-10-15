"""Angular alignment and PEC command-line contracts without optional solvers."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

BEMPP_DIR = Path(__file__).resolve().parents[1] / "bempp"
sys.path.insert(0, str(BEMPP_DIR))
import _bempp_common as common
import run_pec_field_case as field_case


class PNValidationContracts(unittest.TestCase):
    def test_structured_mesh_controls_reach_both_reference_solvers(self):
        config = common.ImpedanceValidationConfig(3.,200.,0.,0.,3,4,"structured",2,3,.2)
        for command in (config.julia_command("mesh-control"), config.bempp_command("mesh-control")):
            self.assertEqual(command[command.index("--nx")+1],"2")
            self.assertEqual(command[command.index("--ny")+1],"3")

    def test_angular_alignment_and_strict_key_arity(self):
        first = {(30.0, 60.0): 3.0, (10.0, 20.0): 1.0}
        second = {(10.0, 20.0): 2.0, (30.0, 60.0): 4.0}
        arrays = common.common_angular_arrays(first, second)
        for actual, expected in zip(arrays, ([10.0, 30.0], [20.0, 60.0], [1.0, 3.0], [2.0, 4.0])):
            np.testing.assert_array_equal(actual, expected)
        for keys in (((1.0, 2.0, 3.0),), ((1.0,),), ((1.0, 2.0), (3.0, 4.0, 5.0))):
            mapping = {key: 1.0 for key in keys}
            with self.assertRaisesRegex(ValueError, "exactly theta and phi"):
                common.common_angular_arrays(mapping, mapping)
        with self.assertRaises(SystemExit):
            common.common_angular_arrays({}, {})

    def test_project_root_controls_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            argv = ["run_pec_field_case.py", "case.json", "nested/report.json",
                    "--project-root", str(root), "--relative-field-tolerance", "0.02"]
            with patch.object(sys, "argv", argv), patch.object(field_case, "solve_case", return_value=0) as solve:
                self.assertEqual(field_case.main(), 0)
            args = solve.call_args[0][0]
            self.assertEqual(args.project_root, root)
            self.assertEqual(args.input, root / "case.json")
            self.assertEqual(args.output, root / "nested" / "report.json")

    def test_absolute_paths_are_not_rebased(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, target = root / "absolute.json", root / "output.json"
            argv = ["run_pec_field_case.py", str(source), str(target),
                    "--project-root", str(BEMPP_DIR), "--relative-field-tolerance", "0.02"]
            with patch.object(sys, "argv", argv), patch.object(field_case, "solve_case", return_value=0) as solve:
                self.assertEqual(field_case.main(), 0)
            args = solve.call_args[0][0]
            self.assertEqual(args.input, source)
            self.assertEqual(args.output, target)

    def test_report_hash_describes_input_read_before_external_solve(self):
        # Mock only the external solver boundary; exercise actual case parsing
        # and report writing while the original input is edited during solve.
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary)/"case.json", Path(temporary)/"report.json"
            case = dict(schema_version=1, convention="exp(+i*omega*t)", case_id="original",
                vertices_m=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]], triangles_zero_based=[[0,1,2],[0,2,3]],
                looks_unit=[[0,0,1]], incident_wavevector_rad_m=[0,0,2*np.pi],
                incident_polarization=[1,0,0], frequency_hz=1, c0_m_s=1,
                incident_amplitude_v_m=1, theta_rad=[0], phi_rad=[0],
                field_real=[[-1,0]], field_imag=[[0,0]], dof_count=1, julia_source_sha256="a"*64)
            source.write_text(json.dumps(case), encoding="utf-8")
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            operator = SimpleNamespace(weak_form=lambda: SimpleNamespace(A=np.ones((1,1))))
            potential = MagicMock()
            potential.__mul__.return_value = np.array([[1],[0],[0]], dtype=complex)
            fake = SimpleNamespace(__version__="test-boundary", Grid=lambda *args: None,
                DefaultParameters=lambda: SimpleNamespace(quadrature=SimpleNamespace()),
                function_space=lambda *args: SimpleNamespace(global_dof_count=1), complex_callable=lambda f:f,
                GridFunction=lambda *args,**kwargs: SimpleNamespace(projections=lambda _:np.ones(1)),
                operators=SimpleNamespace(boundary=SimpleNamespace(maxwell=SimpleNamespace(electric_field=lambda *a,**k:operator)),
                    far_field=SimpleNamespace(maxwell=SimpleNamespace(electric_field=lambda *a,**k:potential))))
            def solve(*args):
                source.write_text(json.dumps(dict(case, case_id="edited")), encoding="utf-8")
                return SimpleNamespace(coefficients=np.ones(1))
            args = SimpleNamespace(input=source, output=target, max_dofs=2, regular_order=4,
                singular_order=6, relative_field_tolerance=.02, absolute_field_tolerance=1e-10)
            with patch.object(field_case,"load_bempp",return_value=(fake,solve)), patch.object(field_case,"version",return_value="test-boundary"):
                self.assertEqual(field_case.solve_case(args),0)
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(report["case_id"],"original")
            self.assertEqual(report["input_sha256"],original_hash)


if __name__ == "__main__":
    unittest.main()
