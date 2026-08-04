"""`tools/q2_preflight.py` — push-button Q-2 GPU-window precondition check
(#62, #228).

Same import pattern as `test_install_script.py`: the tool lives outside
`dgemma`/`surfaces`/`consumers` (a standalone dev tool, not part of the node
pack), so it's imported directly off `tools/` via `sys.path.insert`, not
through the package.

No live GPU is touched — every `subprocess.run` call is monkeypatched via the
tool's own `run=` seam (`check_gpu`, `check_skeleton_branch`,
`collect_environment_provenance` all take an injectable `run` callable
defaulting to the real `subprocess.run` wrapper). This suite always runs
(never SKIPs): it is pure-function coverage over the aggregation/report
shape, not a live-hardware check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
import q2_preflight  # noqa: E402  (path insert must precede this)


class FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run_factory(responses: dict[str, FakeCompletedProcess]):
    """Builds a fake `run(cmd, ...)` that dispatches on `cmd[0]` (the
    executable name) — good enough granularity for this tool's small set of
    subprocess calls (nvidia-smi, git)."""

    def _fake_run(cmd, timeout=None):
        key = cmd[0]
        if key not in responses:
            raise AssertionError(f"unexpected subprocess invocation: {cmd!r}")
        return responses[key]

    return _fake_run


# ---------------------------------------------------------------------------
# check_gpu
# ---------------------------------------------------------------------------


class TestCheckGpu:
    def _run_stub(self, mem_line: str, apps_stdout: str = ""):
        def _fake_run(cmd, timeout=None):
            if cmd[0] == "nvidia-smi" and "--query-gpu=memory.used" in cmd[1]:
                return FakeCompletedProcess(0, stdout=mem_line + "\n")
            if cmd[0] == "nvidia-smi" and "--query-compute-apps" in cmd[1]:
                return FakeCompletedProcess(0, stdout=apps_stdout)
            raise AssertionError(f"unexpected call: {cmd!r}")

        return _fake_run

    def test_nvidia_smi_unreachable_fails(self):
        def _raise(cmd, timeout=None):
            raise FileNotFoundError("nvidia-smi not found")

        results = q2_preflight.check_gpu(run=_raise)
        assert len(results) == 1
        assert results[0].name == "gpu.nvidia_smi_reachable"
        assert results[0].passed is False

    def test_nvidia_smi_nonzero_exit_fails(self):
        def _fake_run(cmd, timeout=None):
            return FakeCompletedProcess(returncode=1, stderr="no devices found")

        results = q2_preflight.check_gpu(run=_fake_run)
        assert results[0].passed is False
        assert "no devices found" in results[0].detail

    def test_sufficient_free_memory_and_only_resident_tenant_passes_clean(self):
        # 1631 used, 46768 free, 49152 total (matches the #226/#62 banked baseline).
        run = self._run_stub(
            mem_line="1631, 46768, 49152",
            apps_stdout=(
                "20276, 110, /usr/lib/xorg/Xorg\n"
                "20546, 9, /usr/bin/gnome-shell\n"
                "63519, 1402, /home/shane/github/llama.cpp/build/bin/llama-server\n"
            ),
        )
        results = q2_preflight.check_gpu(run=run)
        by_name = {r.name: r for r in results}
        assert by_name["gpu.nvidia_smi_reachable"].passed is True
        assert by_name["gpu.free_memory"].passed is True
        assert by_name["gpu.tenancy"].passed is True
        assert by_name["gpu.tenancy"].warning is False

    def test_insufficient_free_memory_fails(self):
        # Only 10 GiB free — below the 35 GiB floor.
        run = self._run_stub(mem_line="39000, 10240, 49152", apps_stdout="")
        results = q2_preflight.check_gpu(run=run)
        by_name = {r.name: r for r in results}
        assert by_name["gpu.free_memory"].passed is False
        assert "10.00 GiB free" in by_name["gpu.free_memory"].detail

    def test_other_tenant_present_warns_and_names_it_but_does_not_fail(self):
        run = self._run_stub(
            mem_line="1631, 46768, 49152",
            apps_stdout="12345, 8000, /usr/bin/some-other-process\n",
        )
        results = q2_preflight.check_gpu(run=run)
        by_name = {r.name: r for r in results}
        tenancy = by_name["gpu.tenancy"]
        assert tenancy.warning is True
        assert "some-other-process" in tenancy.detail
        assert "12345" in tenancy.detail
        # Warnings must not flip aggregate PASS by themselves.
        assert q2_preflight.aggregate_pass(results) is True

    def test_oversized_llama_server_is_named_as_non_resident(self):
        # A llama-server process well above the 2 GiB resident ceiling should
        # still be flagged, not silently accepted just because of its name.
        run = self._run_stub(
            mem_line="1631, 46768, 49152",
            apps_stdout="99999, 20000, /home/shane/github/llama.cpp/build/bin/llama-server\n",
        )
        results = q2_preflight.check_gpu(run=run)
        by_name = {r.name: r for r in results}
        assert by_name["gpu.tenancy"].warning is True
        assert "99999" in by_name["gpu.tenancy"].detail

    def test_two_llama_servers_second_is_flagged_as_non_resident(self):
        # "Known-resident services" (#145 waiver) names ONE llama-server, not
        # an arbitrary count. Two small same-named processes must not both pass
        # as "the resident embedding server" — exactly one is accepted; the
        # second is named as a non-resident tenant (warn, not fail).
        run = self._run_stub(
            mem_line="1631, 46768, 49152",
            apps_stdout=(
                "63519, 1402, /home/shane/github/llama.cpp/build/bin/llama-server\n"
                "63999, 1300, /home/shane/github/llama.cpp/build/bin/llama-server\n"
            ),
        )
        results = q2_preflight.check_gpu(run=run)
        by_name = {r.name: r for r in results}
        tenancy = by_name["gpu.tenancy"]
        # The second llama-server is named (warn-not-fail posture preserved).
        assert tenancy.warning is True
        assert "63999" in tenancy.detail
        # The lower-memory llama-server (pid 63519) is the accepted resident
        # and must NOT be named as an "other" tenant.
        assert "63519" not in tenancy.detail
        # Warn does not flip the aggregate; the >=35 GiB floor is the backstop.
        assert q2_preflight.aggregate_pass(results) is True

    def test_unparseable_memory_line_fails(self):
        def _fake_run(cmd, timeout=None):
            return FakeCompletedProcess(0, stdout="garbage\n")

        results = q2_preflight.check_gpu(run=_fake_run)
        assert results[-1].passed is False


# ---------------------------------------------------------------------------
# check_skeleton_branch
# ---------------------------------------------------------------------------


class TestCheckSkeletonBranch:
    def test_matching_sha_passes(self):
        def _fake_run(cmd, timeout=None):
            return FakeCompletedProcess(
                0,
                stdout=f"{q2_preflight.SKELETON_EXPECTED_SHA}\trefs/heads/"
                f"{q2_preflight.SKELETON_BRANCH}\n",
            )

        result = q2_preflight.check_skeleton_branch(run=_fake_run)
        assert result.passed is True
        assert q2_preflight.SKELETON_EXPECTED_SHA in result.detail

    def test_mismatched_sha_fails(self):
        def _fake_run(cmd, timeout=None):
            return FakeCompletedProcess(
                0, stdout=f"deadbeef1234\trefs/heads/{q2_preflight.SKELETON_BRANCH}\n"
            )

        result = q2_preflight.check_skeleton_branch(run=_fake_run)
        assert result.passed is False
        assert "deadbeef1234" in result.detail
        assert q2_preflight.SKELETON_EXPECTED_SHA in result.detail

    def test_branch_absent_fails(self):
        def _fake_run(cmd, timeout=None):
            return FakeCompletedProcess(0, stdout="")

        result = q2_preflight.check_skeleton_branch(run=_fake_run)
        assert result.passed is False
        assert "no ref" in result.detail

    def test_git_invocation_error_fails(self):
        def _raise(cmd, timeout=None):
            raise OSError("network unreachable")

        result = q2_preflight.check_skeleton_branch(run=_raise)
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_fixture
# ---------------------------------------------------------------------------


class TestCheckFixture:
    def test_present_fixture_passes(self, tmp_path):
        fixture_dir = tmp_path / "examples" / "smoke-tests"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "kv-cache-tier1.api.json").write_text("{}")
        result = q2_preflight.check_fixture(repo_root=tmp_path)
        assert result.passed is True

    def test_missing_fixture_fails(self, tmp_path):
        result = q2_preflight.check_fixture(repo_root=tmp_path)
        assert result.passed is False

    def test_real_repo_fixture_is_present(self):
        """Sanity check against the actual shipped fixture — this task's own
        line 4 precondition."""
        result = q2_preflight.check_fixture(repo_root=REPO_ROOT)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_port_free
# ---------------------------------------------------------------------------


class TestCheckPortFree:
    def test_free_port_passes(self):
        # Port 0 binding trick: ask the OS for an ephemeral free port, then
        # check that exact port before anything binds it.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]

        result = q2_preflight.check_port_free(host="127.0.0.1", port=free_port)
        assert result.passed is True

    def test_occupied_port_fails(self):
        import socket

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            result = q2_preflight.check_port_free(host="127.0.0.1", port=port)
            assert result.passed is False
        finally:
            server.close()


# ---------------------------------------------------------------------------
# collect_environment_provenance
# ---------------------------------------------------------------------------


class TestCollectEnvironmentProvenance:
    def test_banks_driver_and_cuda_from_nvidia_smi(self):
        def _fake_run(cmd, timeout=None):
            if "--query-gpu=driver_version" in cmd:
                return FakeCompletedProcess(0, stdout="580.173.02\n")
            if cmd == ["nvidia-smi"]:
                return FakeCompletedProcess(
                    0,
                    stdout=(
                        "| NVIDIA-SMI 580.173.02 Driver Version: 580.173.02 "
                        "CUDA Version: 13.0 |\n"
                    ),
                )
            raise AssertionError(f"unexpected call: {cmd!r}")

        provenance = q2_preflight.collect_environment_provenance(run=_fake_run)
        assert provenance["driver_version"] == "580.173.02"
        assert provenance["cuda_version_nvidia_smi"] == "13.0"

    def test_missing_nvidia_smi_degrades_to_none_not_raise(self):
        def _raise(cmd, timeout=None):
            raise FileNotFoundError("no nvidia-smi")

        provenance = q2_preflight.collect_environment_provenance(run=_raise)
        assert provenance["driver_version"] is None
        assert provenance["cuda_version_nvidia_smi"] is None
        # torch/transformers/diffusers keys must still be present (may or may
        # not be importable in the test environment — either is fine).
        assert "torch_version" in provenance
        assert "transformers_version" in provenance
        assert "diffusers_version" in provenance


# ---------------------------------------------------------------------------
# aggregate_pass
# ---------------------------------------------------------------------------


class TestAggregatePass:
    def test_all_pass_is_true(self):
        results = [
            q2_preflight.CheckResult("a", True, "ok"),
            q2_preflight.CheckResult("b", True, "ok"),
        ]
        assert q2_preflight.aggregate_pass(results) is True

    def test_one_fail_is_false(self):
        results = [
            q2_preflight.CheckResult("a", True, "ok"),
            q2_preflight.CheckResult("b", False, "nope"),
        ]
        assert q2_preflight.aggregate_pass(results) is False

    def test_warning_failure_does_not_flip_aggregate(self):
        results = [
            q2_preflight.CheckResult("a", True, "ok"),
            q2_preflight.CheckResult("b", False, "warned-but-failed", warning=True),
        ]
        assert q2_preflight.aggregate_pass(results) is True

    def test_empty_results_is_true(self):
        assert q2_preflight.aggregate_pass([]) is True


# ---------------------------------------------------------------------------
# format_report / build_report_dict — report shape
# ---------------------------------------------------------------------------


class TestReportShape:
    def _sample_results(self):
        return [
            q2_preflight.CheckResult("gpu.nvidia_smi_reachable", True, "nvidia-smi responded"),
            q2_preflight.CheckResult("gpu.free_memory", True, "46.00 GiB free"),
            q2_preflight.CheckResult("gpu.tenancy", True, "only resident tenant", warning=False),
            q2_preflight.CheckResult("weights.hf_cache_present", False, "not cached"),
        ]

    def _sample_provenance(self):
        return {
            "driver_version": "580.173.02",
            "cuda_version_nvidia_smi": "13.0",
            "torch_version": "2.12.1+cu130",
            "torch_cuda_version": "13.0",
            "transformers_version": "5.13.0",
            "diffusers_version": "0.39.0",
        }

    def test_format_report_labels_each_line_pass_or_fail(self):
        text = q2_preflight.format_report(
            self._sample_results(), self._sample_provenance(), label="unit-test"
        )
        assert "[PASS] gpu.nvidia_smi_reachable" in text
        assert "[FAIL] weights.hf_cache_present" in text
        assert "Overall: FAIL" in text
        assert q2_preflight.RUNSHEET_URL in text
        assert "unit-test" in text

    def test_format_report_warning_uses_warn_tag_not_pass_or_fail(self):
        results = [q2_preflight.CheckResult("gpu.tenancy", False, "other tenant", warning=True)]
        text = q2_preflight.format_report(results, {}, label="unit-test")
        assert "[WARN] gpu.tenancy" in text
        assert "[FAIL] gpu.tenancy" not in text

    def test_format_report_all_pass_says_overall_pass(self):
        results = [q2_preflight.CheckResult("a", True, "ok")]
        text = q2_preflight.format_report(results, {}, label="unit-test")
        assert "Overall: PASS" in text

    def test_build_report_dict_shape_is_json_serializable(self):
        report = q2_preflight.build_report_dict(
            self._sample_results(), self._sample_provenance(), label="unit-test"
        )
        # Must round-trip through json — this is what --out actually writes.
        text = json.dumps(report)
        reloaded = json.loads(text)

        assert reloaded["label"] == "unit-test"
        assert reloaded["runsheet_url"] == q2_preflight.RUNSHEET_URL
        assert reloaded["overall_pass"] is False  # one FAIL in the sample
        assert len(reloaded["checks"]) == 4
        assert reloaded["checks"][0]["name"] == "gpu.nvidia_smi_reachable"
        assert reloaded["checks"][0]["status"] == "PASS"
        assert reloaded["checks"][-1]["status"] == "FAIL"
        assert reloaded["environment_provenance"]["torch_version"] == "2.12.1+cu130"

    def test_build_report_dict_overall_pass_true_when_all_pass(self):
        results = [q2_preflight.CheckResult("a", True, "ok")]
        report = q2_preflight.build_report_dict(results, {}, label="x")
        assert report["overall_pass"] is True


# ---------------------------------------------------------------------------
# main() — CLI wiring (argv parsing, --out write, exit code)
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_help_mentions_issue_62_amendment_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            q2_preflight.parse_args(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "62" in captured.out
        assert "Amendment 1" in captured.out or "amendment" in captured.out.lower()

    def test_label_is_required(self):
        with pytest.raises(SystemExit):
            q2_preflight.parse_args([])

    def test_out_defaults_to_label_derived_tmp_path(self):
        args = q2_preflight.parse_args(["--label", "my-run"])
        assert args.out is None
        assert args.label == "my-run"

    def test_main_writes_report_and_returns_nonzero_on_any_fail(self, monkeypatch, tmp_path):
        out_path = tmp_path / "report.json"

        def _fake_run_all_checks(repo_root=q2_preflight.REPO_ROOT, run=None):
            return [q2_preflight.CheckResult("weights.hf_cache_present", False, "not cached")]

        def _fake_provenance(run=None):
            return {"torch_version": None}

        monkeypatch.setattr(q2_preflight, "run_all_checks", _fake_run_all_checks)
        monkeypatch.setattr(
            q2_preflight, "collect_environment_provenance", _fake_provenance
        )

        exit_code = q2_preflight.main(["--label", "unit-test", "--out", str(out_path)])

        assert exit_code == 1
        assert out_path.exists()
        written = json.loads(out_path.read_text())
        assert written["overall_pass"] is False
        assert written["label"] == "unit-test"

    def test_main_returns_zero_when_all_pass(self, monkeypatch, tmp_path):
        out_path = tmp_path / "report.json"

        def _fake_run_all_checks(repo_root=q2_preflight.REPO_ROOT, run=None):
            return [q2_preflight.CheckResult("gpu.nvidia_smi_reachable", True, "ok")]

        def _fake_provenance(run=None):
            return {"torch_version": "2.12.1"}

        monkeypatch.setattr(q2_preflight, "run_all_checks", _fake_run_all_checks)
        monkeypatch.setattr(
            q2_preflight, "collect_environment_provenance", _fake_provenance
        )

        exit_code = q2_preflight.main(["--label", "unit-test", "--out", str(out_path)])
        assert exit_code == 0

    def test_main_default_out_path_uses_label_no_timestamp(self, monkeypatch, tmp_path):
        """The task's line 6: default path is /tmp/q2-preflight-<label> with
        NO auto-generated timestamp — the label alone determines the path."""
        monkeypatch.setattr(
            q2_preflight,
            "run_all_checks",
            lambda repo_root=q2_preflight.REPO_ROOT, run=None: [
                q2_preflight.CheckResult("a", True, "ok")
            ],
        )
        monkeypatch.setattr(
            q2_preflight, "collect_environment_provenance", lambda run=None: {}
        )

        # Redirect /tmp writes into tmp_path by monkeypatching Path directly
        # would be invasive; instead assert the *name* the tool derives,
        # then clean up if it actually landed in /tmp.
        exit_code = q2_preflight.main(["--label", "pytest-unit-test-marker"])
        expected_path = Path("/tmp/q2-preflight-pytest-unit-test-marker.json")
        try:
            assert exit_code == 0
            assert expected_path.exists()
            content = json.loads(expected_path.read_text())
            assert content["label"] == "pytest-unit-test-marker"
        finally:
            expected_path.unlink(missing_ok=True)
