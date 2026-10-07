"""Command deadlines preserve exit status, evidence and ownership checks."""

import io
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from slopfix_lib import quality


class QualityCommandDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name

    def gate(self, timeout=None):
        return quality._command_gate(
            "deadline-test", "functional-suitability", "Command contract",
            [sys.executable, "-c", "pass"], timeout_seconds=timeout,
        )

    def test_null_and_explicit_platform_bounded_deadlines_validate(self):
        for timeout in (None, 1, int(threading.TIMEOUT_MAX)):
            with self.subTest(timeout=timeout):
                quality._validate_common(self.gate(timeout), self.root, 0)
        for timeout in (False, True, 0, -1, 1.5, "1", int(threading.TIMEOUT_MAX)+1):
            with self.subTest(timeout=timeout):
                with self.assertRaises(quality.QualityConfigError):
                    quality._validate_common(self.gate(timeout), self.root, 0)

    def test_missing_deadline_requires_an_explicit_choice(self):
        gate = self.gate()
        del gate["timeout_seconds"]
        with self.assertRaisesRegex(quality.QualityConfigError, "timeout_seconds"):
            quality._validate_common(gate, self.root, 0)

    def test_julia_template_does_not_invent_command_budgets(self):
        pathlib.Path(self.root, "Project.toml").write_text('name = "DeadlineProbe"\n', encoding="utf-8")
        config = quality.build(self.root, "julia")
        commands = [g for g in config["gates"] if g["kind"] == "command"]
        self.assertTrue(commands)
        self.assertTrue(all(g["timeout_seconds"] is None for g in commands))
        self.assertTrue(all(g["required"] for g in commands))

    def run_owned_mock(self, gate, *, returncode=0, timeout=False, descendants=False, mutation=None):
        process = types.SimpleNamespace(
            stdout=io.BytesIO(b"complete output\n"), stderr=io.BytesIO(b"diagnostic output\n"),
            pid=os.getpid(), wait=mock.Mock(),
        )

        def wait(**kwargs):
            if mutation is not None:
                mutation()
            if timeout:
                raise subprocess.TimeoutExpired(gate["command"], gate["timeout_seconds"])
            return returncode

        process.wait.side_effect = wait
        # Simulate the POSIX runner boundary without changing global os.name
        # or claiming Windows supports process-group cleanup.
        platform = types.SimpleNamespace(name="posix", path=os.path, environ=os.environ, sep=os.sep)
        with mock.patch.object(quality, "os", platform), \
             mock.patch.object(quality.subprocess, "Popen", return_value=process) as launch, \
             mock.patch.object(quality, "_terminate_owned_process_tree", return_value=descendants) as cleanup:
            result = quality._run_command(gate, self.root)
        launch.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        process.wait.assert_called_once_with(timeout=gate["timeout_seconds"])
        cleanup.assert_called_once_with(process)
        self.assertEqual(result["stdout"]["bytes"], len(b"complete output\n"))
        self.assertEqual(result["stderr"]["bytes"], len(b"diagnostic output\n"))
        return result

    def test_null_wait_retains_success_and_stream_evidence(self):
        result = self.run_owned_mock(self.gate())
        self.assertEqual(result["status"], quality.PASS)
        self.assertEqual(result["exit_code"], 0)

    def test_null_wait_preserves_nonzero_exit_failure(self):
        result = self.run_owned_mock(self.gate(), returncode=1)
        self.assertEqual(result["status"], quality.FAIL)
        self.assertEqual(result["exit_code"], 1)

    def test_explicit_deadline_still_fails_and_cleans_exact_owned_process(self):
        result = self.run_owned_mock(self.gate(1), timeout=True)
        self.assertEqual(result["status"], quality.FAIL)
        self.assertEqual(result["message"], "timed out after 1 seconds")

    def test_null_wait_rejects_surviving_descendants(self):
        result = self.run_owned_mock(self.gate(), descendants=True)
        self.assertEqual(result["status"], quality.FAIL)
        self.assertIn("surviving descendant", result["message"])

    def test_null_wait_rejects_protected_file_mutation(self):
        path = pathlib.Path(self.root, "protected.txt")
        path.write_bytes(b"before")
        gate = self.gate()
        gate["protect_paths"] = [path.name]
        result = self.run_owned_mock(gate, mutation=lambda: path.write_bytes(b"after"))
        self.assertEqual(result["status"], quality.FAIL)
        self.assertEqual(result["protected_paths_changed"], [path.name])


if __name__ == "__main__":
    unittest.main()
