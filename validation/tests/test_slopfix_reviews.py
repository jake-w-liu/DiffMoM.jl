"""Reviewed detector findings must remain exact and visible without changing counts."""

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from slopfix_lib import counting, langs, manifest, reviews, scope, smells


class IntegrityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        (self.root / ".slopfix").mkdir()

    def record(self, name, text, rule, **kwargs):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
        return dict(path=name, rule=rule, hash_mode="raw",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    reason="Reviewed exact input boundary or immutable primary evidence.", **kwargs)

    def write_reviews(self, entries):
        (self.root / ".slopfix/integrity-reviews.json").write_text(
            json.dumps(dict(schema_version=1, reviews=entries)), encoding="utf-8")

    def test_exact_smell_stays_visible_as_advisory(self):
        text = 'p[4]=="-1" && throw(ArgumentError("resize is not implemented"))\n'
        entry = self.record("source.jl", text, "placeholder-implementation", line=1)
        self.write_reviews([entry])
        hit = next(h for h in smells.scan_text("source.jl", text, langs.detect("source.jl"))
                   if h.severity == smells.BLOCKING)
        cleared = reviews.IntegrityReviews(str(self.root)).apply(hit)
        self.assertEqual(cleared.severity, smells.ADVISORY)
        self.assertEqual(cleared.excerpt, hit.excerpt)
        self.assertIn(entry["reason"], cleared.message)
        self.assertEqual(hit.severity, smells.BLOCKING)
        new = smells.Hit("other.jl", 1, hit.rule, hit.severity, hit.message, hit.excerpt)
        self.assertIs(reviews.IntegrityReviews(str(self.root)).apply(new), new)

    def test_long_fixture_still_counts_and_live_long_code_still_flags(self):
        name = "test/fixtures/manual.html"
        text = "<p>" + "native primary documentation " * 40 + "</p>\n"
        entry = self.record(name, text, "long-line-introduced")
        self.write_reviews([entry])
        live = self.root / "source.jl"
        live.write_text("value = 1\n", encoding="utf-8")
        sc = scope.Scope(root=str(self.root))
        files = [(name, langs.detect(name)), ("source.jl", langs.detect("source.jl"))]
        before = manifest._builtin_metrics(str(self.root), sc, files)
        checked = reviews.IntegrityReviews(str(self.root))
        after = manifest._builtin_metrics(str(self.root), sc, files, checked.long_lines)
        self.assertEqual(before["code_lines"], after["code_lines"])
        self.assertEqual(before["max_code_line"], after["max_code_line"])
        self.assertGreater(after["max_code_line"], 1000)
        self.assertEqual(after["max_unreviewed_code_line"], len("value = 1"))
        live.write_text("value = " + "1 + " * 80 + "0\n", encoding="utf-8")
        changed = manifest._builtin_metrics(str(self.root), sc, files, checked.long_lines)
        self.assertGreater(changed["max_unreviewed_code_line"], 274)

    def test_changed_or_missing_reviewed_source_fails(self):
        entry = self.record("test/fixtures/manual.html", "<p>source</p>\n", "long-line-introduced")
        self.write_reviews([entry])
        path = self.root / entry["path"]
        path.write_text("<p>edited</p>\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            reviews.IntegrityReviews(str(self.root))
        path.unlink()
        with self.assertRaises(OSError):
            reviews.IntegrityReviews(str(self.root))

    def test_only_explicit_lf_mode_normalizes_line_endings(self):
        entry = self.record("source.jl", 'throw(ArgumentError("not implemented"))\n',
                            "placeholder-implementation", line=1)
        entry["hash_mode"] = "lf"
        self.write_reviews([entry])
        path = self.root / entry["path"]
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        reviews.IntegrityReviews(str(self.root))
        entry["hash_mode"] = "raw"
        self.write_reviews([entry])
        with self.assertRaisesRegex(ValueError, "stale"):
            reviews.IntegrityReviews(str(self.root))

    def test_bad_review_records_fail_closed(self):
        entry = self.record("source.jl", 'throw(ArgumentError("not implemented"))\n',
                            "placeholder-implementation", line=1)
        for mutation in ({"path": "../outside.jl"}, {"sha256": "bad"}, {"line": True},
                         {"line": []}, {"line": 0}, {"line": 2}, {"rule": "swallowed-error"},
                         {"hash_mode": "unknown"}, {"reason": ""}):
            with self.subTest(mutation=mutation):
                self.write_reviews([dict(entry, **mutation)])
                with self.assertRaises((ValueError, OSError)):
                    reviews.IntegrityReviews(str(self.root))
        self.write_reviews([entry, entry])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reviews.IntegrityReviews(str(self.root))
        active = self.record("active.html", "<p>active</p>\n", "long-line-introduced")
        self.write_reviews([active])
        with self.assertRaisesRegex(ValueError, "archival"):
            reviews.IntegrityReviews(str(self.root))

    def test_missing_policy_never_approves_a_finding(self):
        policy = reviews.IntegrityReviews(str(self.root))
        hit = smells.Hit("source.jl", 1, "placeholder-implementation", smells.BLOCKING,
                         "Unimplemented", 'throw(ArgumentError("not implemented"))')
        self.assertIs(policy.apply(hit), hit)
        self.assertFalse(policy.long_lines)


if __name__ == "__main__":
    unittest.main()
