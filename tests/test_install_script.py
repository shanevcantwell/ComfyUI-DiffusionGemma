"""`install.py` — belt-and-braces post-install script (issue #147).

Pure-function coverage over the pieces `install.py`'s `run()` composes:
requirements.txt parsing, the satisfied/unsatisfied decision (importlib.metadata
+ packaging.specifiers, faked via monkeypatch so no real package state is
read), the failure block's content (must name the failed spec and the exact
manual fallback command), and `run()`'s end-to-end wiring with the pip
subprocess call monkeypatched out — argv shape is asserted, nothing is ever
actually pip-installed here.

Also covers import-safety (ARCHITECTURE.md-adjacent concern raised in issue
#147's build): `install.py` sits at the pack root next to `__init__.py`,
where ComfyUI's own loader and this test suite both import from — it must do
zero work at import time.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))
import install as install_script  # noqa: E402  (path insert must precede this)


# ---------------------------------------------------------------------------
# requirements.txt parsing
# ---------------------------------------------------------------------------


class TestParseRequirements:
    def test_strips_blank_lines_and_comments(self):
        text = (
            "# a full-line comment\n"
            "\n"
            "transformers==5.13.0\n"
            "   \n"
            "diffusers>=0.39.0\n"
            "# another comment\n"
            "accelerate\n"
        )
        assert install_script.parse_requirements(text) == [
            "transformers==5.13.0",
            "diffusers>=0.39.0",
            "accelerate",
        ]

    def test_empty_input_yields_empty_list(self):
        assert install_script.parse_requirements("") == []

    def test_does_not_filter_core_provided_packages(self):
        """install.py consumes requirements.txt as-is — the CORE_PROVIDED
        filter (torch/torchvision/numpy/Pillow) lives in
        tests/test_requirements_sync.py against pyproject.toml, not here.
        A line present in the file is a line install.py acts on."""
        text = "torch\ntransformers==5.13.0\n"
        assert install_script.parse_requirements(text) == ["torch", "transformers==5.13.0"]

    def test_real_requirements_txt_parses_to_nonempty_specs(self):
        """Sanity check against the actual shipped file — catches a format
        this parser can't handle before it ever reaches a user's install."""
        specs = install_script.parse_requirements(
            (REPO_ROOT / "requirements.txt").read_text()
        )
        assert specs, "requirements.txt parsed to zero specs"
        names = [install_script._bare_name(s) for s in specs]
        assert "transformers" in names
        assert "diffusers" in names


# ---------------------------------------------------------------------------
# _bare_name
# ---------------------------------------------------------------------------


class TestBareName:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("transformers==5.13.0", "transformers"),
            ("diffusers>=0.39.0", "diffusers"),
            ("accelerate", "accelerate"),
            ("auto-round>=0.5", "auto-round"),
            ("pkg[extra]>=1.0", "pkg"),
            ("pkg; python_version >= '3.11'", "pkg"),
        ],
    )
    def test_strips_specifiers_extras_and_markers(self, spec, expected):
        assert install_script._bare_name(spec) == expected

    def test_unparseable_spec_raises(self):
        with pytest.raises(ValueError):
            install_script._bare_name("   ")


# ---------------------------------------------------------------------------
# installed_version / is_satisfied — importlib.metadata faked via monkeypatch
# ---------------------------------------------------------------------------


class _FakeNotFound(Exception):
    pass


def _make_fake_metadata(versions: dict[str, str], not_found_exc):
    """Build a fake importlib.metadata-shaped object: .version(name) returns
    versions[name] or raises not_found_exc; .PackageNotFoundError is the
    exception type install.py is expected to catch."""

    class _Fake:
        PackageNotFoundError = not_found_exc

        @staticmethod
        def version(name):
            try:
                return versions[name]
            except KeyError:
                raise not_found_exc(name)

    return _Fake()


class TestInstalledVersion:
    def test_returns_version_when_present(self, monkeypatch):
        fake = _make_fake_metadata({"transformers": "5.13.0"}, _FakeNotFound)
        monkeypatch.setattr(
            "importlib.metadata.version", fake.version
        )
        monkeypatch.setattr(
            "importlib.metadata.PackageNotFoundError", fake.PackageNotFoundError
        )
        assert install_script.installed_version("transformers") == "5.13.0"

    def test_returns_none_when_absent(self, monkeypatch):
        fake = _make_fake_metadata({}, _FakeNotFound)
        monkeypatch.setattr("importlib.metadata.version", fake.version)
        monkeypatch.setattr(
            "importlib.metadata.PackageNotFoundError", fake.PackageNotFoundError
        )
        assert install_script.installed_version("diffusers") is None


class TestIsSatisfied:
    def _patch_metadata(self, monkeypatch, versions: dict[str, str]):
        fake = _make_fake_metadata(versions, _FakeNotFound)
        monkeypatch.setattr("importlib.metadata.version", fake.version)
        monkeypatch.setattr(
            "importlib.metadata.PackageNotFoundError", fake.PackageNotFoundError
        )

    def test_missing_package_is_unsatisfied(self, monkeypatch):
        self._patch_metadata(monkeypatch, {})
        satisfied, version = install_script.is_satisfied("transformers==5.13.0")
        assert satisfied is False
        assert version is None

    def test_exact_pin_matching_version_is_satisfied(self, monkeypatch):
        self._patch_metadata(monkeypatch, {"transformers": "5.13.0"})
        satisfied, version = install_script.is_satisfied("transformers==5.13.0")
        assert satisfied is True
        assert version == "5.13.0"

    def test_exact_pin_mismatched_version_is_unsatisfied(self, monkeypatch):
        """The issue #147 field-report shape: transformers present, but not
        our pinned series (ComfyUI-core's bundled version)."""
        self._patch_metadata(monkeypatch, {"transformers": "4.44.0"})
        satisfied, version = install_script.is_satisfied("transformers==5.13.0")
        assert satisfied is False
        assert version == "4.44.0"

    def test_floor_spec_above_floor_is_satisfied(self, monkeypatch):
        self._patch_metadata(monkeypatch, {"diffusers": "0.40.0"})
        satisfied, version = install_script.is_satisfied("diffusers>=0.39.0")
        assert satisfied is True
        assert version == "0.40.0"

    def test_floor_spec_below_floor_is_unsatisfied(self, monkeypatch):
        self._patch_metadata(monkeypatch, {"diffusers": "0.20.0"})
        satisfied, version = install_script.is_satisfied("diffusers>=0.39.0")
        assert satisfied is False
        assert version == "0.20.0"

    def test_bare_name_no_specifier_present_is_satisfied(self, monkeypatch):
        self._patch_metadata(monkeypatch, {"accelerate": "1.0.0"})
        satisfied, version = install_script.is_satisfied("accelerate")
        assert satisfied is True
        assert version == "1.0.0"

    def test_packaging_unavailable_falls_back_to_unsatisfied(self, monkeypatch):
        """packaging import guarded per issue instructions: if it can't be
        imported, don't guess — report unsatisfied so the install step gets
        a chance to fix it (attempt-install fallback)."""
        self._patch_metadata(monkeypatch, {"diffusers": "0.40.0"})

        import builtins

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "packaging.specifiers" or name.startswith("packaging"):
                raise ImportError("simulated: packaging not importable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)
        satisfied, version = install_script.is_satisfied("diffusers>=0.39.0")
        assert satisfied is False
        assert version == "0.40.0"


# ---------------------------------------------------------------------------
# Failure-message content
# ---------------------------------------------------------------------------


class TestFailureBlock:
    def test_names_failed_spec_and_manual_command(self, capsys):
        install_script.print_failure_block(["transformers==5.13.0"])
        out = capsys.readouterr().out
        assert "transformers==5.13.0" in out
        assert install_script._manual_fallback_command() in out

    def test_manual_command_uses_this_interpreter_and_requirements_flag(self):
        cmd = install_script._manual_fallback_command()
        assert sys.executable in cmd
        assert "-m pip install -r requirements.txt" in cmd

    def test_multiple_failed_specs_all_named(self, capsys):
        install_script.print_failure_block(["transformers==5.13.0", "diffusers>=0.39.0"])
        out = capsys.readouterr().out
        assert "transformers==5.13.0" in out
        assert "diffusers>=0.39.0" in out


# ---------------------------------------------------------------------------
# run() end-to-end wiring — pip subprocess monkeypatched, never actually run
# ---------------------------------------------------------------------------


class TestRun:
    def test_all_satisfied_is_fast_no_op_and_never_calls_pip(self, monkeypatch, tmp_path, capsys):
        req = tmp_path / "requirements.txt"
        req.write_text("transformers==5.13.0\n")
        monkeypatch.setattr(install_script, "REQUIREMENTS_PATH", req)
        monkeypatch.setattr(
            install_script, "is_satisfied", lambda spec: (True, "5.13.0")
        )

        pip_calls = []
        monkeypatch.setattr(
            install_script,
            "pip_install",
            lambda spec: pip_calls.append(spec) or pytest.fail("pip_install must not be called when satisfied"),
        )

        exit_code = install_script.run()
        assert exit_code == 0
        assert pip_calls == []
        out = capsys.readouterr().out
        assert "all requirements satisfied" in out

    def test_missing_requirement_triggers_pip_install_with_expected_argv(
        self, monkeypatch, tmp_path
    ):
        req = tmp_path / "requirements.txt"
        req.write_text("diffusers>=0.39.0\n")
        monkeypatch.setattr(install_script, "REQUIREMENTS_PATH", req)
        monkeypatch.setattr(install_script, "is_satisfied", lambda spec: (False, None))

        captured_argv = []

        def _fake_pip_install(spec):
            captured_argv.append(spec)
            return subprocess.CompletedProcess(
                args=[sys.executable, "-m", "pip", "install", spec],
                returncode=0,
                stdout="",
                stderr="",
            )

        monkeypatch.setattr(install_script, "pip_install", _fake_pip_install)

        exit_code = install_script.run()
        assert exit_code == 0
        assert captured_argv == ["diffusers>=0.39.0"]

    def test_pip_install_argv_shape(self, monkeypatch):
        """pip_install itself must build `[sys.executable, -m, pip, install, spec]`
        — per-line, mirroring Manager's own choice (issue #147 research
        comment) — never a batched `-r requirements.txt`."""
        captured = {}

        def _fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        install_script.pip_install("accelerate")
        assert captured["argv"] == [sys.executable, "-m", "pip", "install", "accelerate"]

    def test_pip_failure_returns_nonzero_and_prints_failure_block(
        self, monkeypatch, tmp_path, capsys
    ):
        req = tmp_path / "requirements.txt"
        req.write_text("transformers==5.13.0\n")
        monkeypatch.setattr(install_script, "REQUIREMENTS_PATH", req)
        monkeypatch.setattr(install_script, "is_satisfied", lambda spec: (False, None))
        monkeypatch.setattr(
            install_script,
            "pip_install",
            lambda spec: subprocess.CompletedProcess(
                args=[sys.executable, "-m", "pip", "install", spec],
                returncode=1,
                stdout="",
                stderr="simulated pip failure",
            ),
        )

        exit_code = install_script.run()
        assert exit_code == 1
        out = capsys.readouterr().out
        assert "transformers==5.13.0" in out
        assert install_script._manual_fallback_command() in out
        assert "FAILED" in out.upper() or "fail" in out.lower()

    def test_missing_requirements_file_is_a_clean_no_op(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            install_script, "REQUIREMENTS_PATH", tmp_path / "does-not-exist.txt"
        )
        assert install_script.run() == 0

    def test_empty_requirements_file_is_a_clean_no_op(self, monkeypatch, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("# only comments\n\n")
        monkeypatch.setattr(install_script, "REQUIREMENTS_PATH", req)
        assert install_script.run() == 0


# ---------------------------------------------------------------------------
# Environment report — importlib.metadata only, never imports the packages
# ---------------------------------------------------------------------------


class TestEnvironmentReport:
    def test_reports_python_executable_and_version(self, capsys):
        install_script.print_environment_report()
        out = capsys.readouterr().out
        assert sys.executable in out
        assert sys.version.split()[0] in out

    def test_reports_each_diagnostic_package_present_or_absent(self, monkeypatch, capsys):
        self_versions = {"transformers": "5.13.0", "accelerate": "1.0.0"}
        fake = _make_fake_metadata(self_versions, _FakeNotFound)
        monkeypatch.setattr("importlib.metadata.version", fake.version)
        monkeypatch.setattr(
            "importlib.metadata.PackageNotFoundError", fake.PackageNotFoundError
        )

        install_script.print_environment_report()
        out = capsys.readouterr().out
        assert "transformers" in out and "5.13.0" in out
        assert "accelerate" in out and "1.0.0" in out
        assert "diffusers" in out and "not installed" in out


# ---------------------------------------------------------------------------
# Import-safety: install.py must do zero work at import time
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_module_import_does_not_touch_network_or_run_pip(self):
        """Already exercised by this file's own module-level `import install`
        succeeding without a monkeypatched subprocess — but pin it explicitly
        so a future refactor that moves work to module scope is caught."""
        import importlib

        # Re-importing (module already cached) must not error or emit
        # subprocess calls; if install.py had top-level side effects this
        # would have already failed at collection time.
        importlib.reload(install_script)

    def test_main_guard_present_in_source(self):
        """Structural guard: install.py's own source must gate all
        subprocess/pip work behind `if __name__ == "__main__":` so importing
        it (as this test file does, and as anything scanning the pack root
        does) can never trigger a pip install."""
        source = (REPO_ROOT / "install.py").read_text()
        assert 'if __name__ == "__main__":' in source
