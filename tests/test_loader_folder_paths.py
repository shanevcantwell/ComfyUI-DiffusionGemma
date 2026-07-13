"""nodes/loader.py's `folder_paths` scanning/resolution glue (issue #17).

`folder_paths` is a ComfyUI-runtime module, genuinely absent in this venv
(`nodes.loader.folder_paths is None` — see the narrow `try/except ImportError`
at the top of `nodes/loader.py`, the same treatment `dgemma/model.py` already
gives the `transformers` import). Every test here injects a fake `folder_paths`
module object via `monkeypatch.setattr("nodes.loader.folder_paths", ...)` so
`list_local_model_dirs`/`resolve_local_model_dir` are exercised as they would
run inside a live ComfyUI process, without needing ComfyUI installed.

Layout convention used throughout: a "valid model dir" is a directory
containing `config.json` (the sentinel — see `nodes/loader.py`'s own
docstring for why: transformers checkpoints are shard directories, so
`folder_paths.get_filename_list`/`get_full_path` — both file-oriented — can't
enumerate or resolve them).
"""
from __future__ import annotations

import os

import pytest

import nodes.loader as loader_module


class FakeFolderPaths:
    """Minimal stand-in for the real `folder_paths` module: only
    `get_folder_paths` is used by `list_local_model_dirs`/
    `resolve_local_model_dir` — no `get_filename_list`/`get_full_path`,
    since those are file-oriented and cannot address a shard directory."""

    def __init__(self, roots_by_key: dict[str, list[str]]):
        self._roots_by_key = roots_by_key

    def get_folder_paths(self, folder_name: str) -> list[str]:
        if folder_name not in self._roots_by_key:
            raise KeyError(folder_name)
        return self._roots_by_key[folder_name]


def _make_model_dir(tmp_path, *parts: str) -> str:
    """Creates <tmp_path>/<parts...>/config.json and returns the dir path."""
    d = tmp_path.joinpath(*parts)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_text("fake shard")
    return str(d)


def _make_bare_dir(tmp_path, *parts: str) -> str:
    """A directory with no config.json — must NOT be treated as a model."""
    d = tmp_path.joinpath(*parts)
    d.mkdir(parents=True)
    return str(d)


class TestListLocalModelDirs:
    def test_returns_empty_list_when_folder_paths_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(loader_module, "folder_paths", None)
        assert loader_module.list_local_model_dirs() == []

    def test_lists_model_dirs_from_diffusion_models_folder(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        _make_model_dir(tmp_path, "diffusion_models", "diffusiongemma-26b")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == ["diffusiongemma-26b"]

    def test_lists_model_dirs_from_text_encoders_folder(self, tmp_path, monkeypatch):
        te_root = tmp_path / "text_encoders"
        te_root.mkdir()
        _make_model_dir(tmp_path, "text_encoders", "diffusiongemma-26b")

        fake = FakeFolderPaths({"diffusion_models": [], "text_encoders": [str(te_root)]})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == ["diffusiongemma-26b"]

    def test_unions_both_folders_not_just_one(self, tmp_path, monkeypatch):
        """Issue #17's core acceptance criterion: BOTH folders are scanned
        and their results unioned — a model under either shows up."""
        dm_root = tmp_path / "diffusion_models"
        te_root = tmp_path / "text_encoders"
        dm_root.mkdir()
        te_root.mkdir()
        _make_model_dir(tmp_path, "diffusion_models", "model-a")
        _make_model_dir(tmp_path, "text_encoders", "model-b")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": [str(te_root)]})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == ["model-a", "model-b"]

    def test_deduplicates_a_name_present_under_both_folders(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        te_root = tmp_path / "text_encoders"
        dm_root.mkdir()
        te_root.mkdir()
        _make_model_dir(tmp_path, "diffusion_models", "same-name")
        _make_model_dir(tmp_path, "text_encoders", "same-name")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": [str(te_root)]})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == ["same-name"]

    def test_ignores_directories_without_the_config_json_sentinel(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        _make_model_dir(tmp_path, "diffusion_models", "real-model")
        _make_bare_dir(tmp_path, "diffusion_models", "not-a-model-just-a-folder")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == ["real-model"]

    def test_ignores_plain_files_at_the_top_level(self, tmp_path, monkeypatch):
        """A loose file (not a directory) directly under the models root must
        never appear in the dropdown — only immediate subdirectories are
        candidates."""
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        (dm_root / "stray.safetensors").write_text("not a shard dir")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == []

    def test_missing_configured_root_directory_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        """A configured root that doesn't exist on disk (never created, or a
        stale ComfyUI config entry) must not raise."""
        nonexistent = str(tmp_path / "does-not-exist")
        fake = FakeFolderPaths({"diffusion_models": [nonexistent], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == []

    def test_unregistered_folder_key_is_skipped_not_fatal(self, monkeypatch):
        """An older ComfyUI without a `text_encoders` folder key registered
        raises KeyError from `get_folder_paths` — must be tolerated, not
        propagated."""

        class RaisingFolderPaths:
            def get_folder_paths(self, folder_name):
                raise KeyError(folder_name)

        monkeypatch.setattr(loader_module, "folder_paths", RaisingFolderPaths())
        assert loader_module.list_local_model_dirs() == []

    def test_result_is_sorted(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        _make_model_dir(tmp_path, "diffusion_models", "zeta-model")
        _make_model_dir(tmp_path, "diffusion_models", "alpha-model")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.list_local_model_dirs() == ["alpha-model", "zeta-model"]


class TestResolveLocalModelDir:
    def test_resolves_name_to_full_path_in_diffusion_models(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        expected = _make_model_dir(tmp_path, "diffusion_models", "diffusiongemma-26b")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("diffusiongemma-26b") == expected

    def test_resolves_name_to_full_path_in_text_encoders(self, tmp_path, monkeypatch):
        te_root = tmp_path / "text_encoders"
        te_root.mkdir()
        expected = _make_model_dir(tmp_path, "text_encoders", "diffusiongemma-26b")

        fake = FakeFolderPaths({"diffusion_models": [], "text_encoders": [str(te_root)]})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("diffusiongemma-26b") == expected

    def test_returns_none_for_unresolvable_name(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("nonexistent-model") is None

    def test_returns_none_for_empty_name(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("") is None

    def test_returns_none_when_folder_paths_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(loader_module, "folder_paths", None)
        assert loader_module.resolve_local_model_dir("anything") is None

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "/etc/passwd",
            "../../../etc/passwd",
            "..",
            "sub/dir",
            "a/../../b",
        ],
    )
    def test_rejects_names_that_are_not_a_single_path_component(self, tmp_path, monkeypatch, malicious_name):
        """Security: ComfyUI's COMBO widget type is UI-only guidance, not a
        server-enforced enum — a `/prompt` POST can send any string for
        `model_name`. Without this guard, `os.path.join(root, name)` for an
        absolute `name` silently discards `root` (documented os.path.join
        behavior) and a `../`-laden `name` walks out of the configured model
        folder, handing an attacker-chosen directory straight to
        `from_pretrained(local_files_only=True)`. Also plants a real
        `config.json` outside the configured root (at `tmp_path/etc`) so a
        naive join-and-isdir check WOULD have resolved it if the guard were
        absent — proving this is a behavioral rejection, not just an absence
        of a matching file."""
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        # A real, loadable model dir OUTSIDE the configured root — proves the
        # guard blocks escape even when the escape target is "valid".
        _make_model_dir(tmp_path, "etc")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir(malicious_name) is None

    def test_accepts_a_normal_single_component_name(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        expected = _make_model_dir(tmp_path, "diffusion_models", "normal-model-name")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("normal-model-name") == expected

    def test_returns_none_for_a_directory_missing_the_config_json_sentinel(self, tmp_path, monkeypatch):
        """A stray non-model directory must not resolve just because its name
        matches — the sentinel check applies on resolution too, not only on
        listing (defends against a directory being emptied/corrupted between
        the dropdown being populated and the node executing)."""
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        _make_bare_dir(tmp_path, "diffusion_models", "not-a-model")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("not-a-model") is None

    def test_unregistered_folder_key_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        """Same tolerance as `list_local_model_dirs` (an older ComfyUI without
        a `diffusion_models` key raises KeyError from `get_folder_paths`), but
        exercised through `resolve_local_model_dir` — the other of the two
        functions that calls `get_folder_paths` per key. The raising key
        (`diffusion_models`) is checked FIRST per `_MODEL_FOLDER_KEYS`'
        ordering, so this also proves the loop continues past the KeyError to
        the next key rather than aborting resolution entirely."""
        expected = _make_model_dir(tmp_path, "text_encoders", "diffusiongemma-26b")

        class PartiallyRaisingFolderPaths:
            def get_folder_paths(self, folder_name):
                if folder_name == "diffusion_models":
                    raise KeyError(folder_name)
                return [str(tmp_path / "text_encoders")]

        monkeypatch.setattr(loader_module, "folder_paths", PartiallyRaisingFolderPaths())

        assert loader_module.resolve_local_model_dir("diffusiongemma-26b") == expected

    def test_diffusion_models_checked_before_text_encoders_on_name_collision(self, tmp_path, monkeypatch):
        """Both folders can carry the same directory name (issue #17 union) —
        resolution must be deterministic. `_MODEL_FOLDER_KEYS` orders
        diffusion_models first, so that copy wins on a collision."""
        dm_expected = _make_model_dir(tmp_path, "diffusion_models", "same-name")
        _make_model_dir(tmp_path, "text_encoders", "same-name")

        fake = FakeFolderPaths(
            {
                "diffusion_models": [str(tmp_path / "diffusion_models")],
                "text_encoders": [str(tmp_path / "text_encoders")],
            }
        )
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        assert loader_module.resolve_local_model_dir("same-name") == dm_expected


class TestLoaderIntegrationWithFakeFolderPaths:
    """End-to-end through `DGemmaLoader.load()` with a fake `folder_paths` and
    a monkeypatched `load_model`, proving the dropdown-to-load_model wiring
    end to end (not just the two helper functions in isolation)."""

    def test_load_resolves_and_forwards_local_files_only_true(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        expected_path = _make_model_dir(tmp_path, "diffusion_models", "diffusiongemma-26b")

        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        captured = {}

        def fake_load_model(repo_id, quant, local_files_only):
            captured["repo_id"] = repo_id
            captured["quant"] = quant
            captured["local_files_only"] = local_files_only
            return "the-loaded-model"

        monkeypatch.setattr(loader_module, "load_model", fake_load_model)

        node = loader_module.DGemmaLoader()
        result = node.load(model_name="diffusiongemma-26b", quant="none")

        assert result == ("the-loaded-model",)
        assert captured["repo_id"] == expected_path
        assert captured["local_files_only"] is True

    def test_load_raises_and_never_calls_load_model_for_missing_selection(self, tmp_path, monkeypatch):
        dm_root = tmp_path / "diffusion_models"
        dm_root.mkdir()
        fake = FakeFolderPaths({"diffusion_models": [str(dm_root)], "text_encoders": []})
        monkeypatch.setattr(loader_module, "folder_paths", fake)

        load_model_calls = []
        monkeypatch.setattr(loader_module, "load_model", lambda **kw: load_model_calls.append(kw))

        node = loader_module.DGemmaLoader()
        with pytest.raises(RuntimeError, match="never falls back to a network fetch"):
            node.load(model_name="nothing-here", quant="none")

        assert load_model_calls == []
