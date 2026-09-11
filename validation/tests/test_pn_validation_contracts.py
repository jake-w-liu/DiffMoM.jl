"""Angular alignment and PEC command-line contracts without optional solvers."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

BEMPP_DIR = Path(__file__).resolve().parents[1] / "bempp"
sys.path.insert(0, str(BEMPP_DIR))
import _bempp_common as common
import run_pec_field_case as field_case


class PNValidationContracts(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
