from __future__ import annotations

import importlib
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

airflow_git_sync = importlib.import_module("airflow_support.git_sync")
GitHubRepo = airflow_git_sync.GitHubRepo
build_github_remote_url = airflow_git_sync.build_github_remote_url
dvc_tracked_paths = airflow_git_sync.dvc_tracked_paths
stage_dataset_path = airflow_git_sync.stage_dataset_path
run_command = airflow_git_sync._run
build_force_with_lease_arg = airflow_git_sync._build_force_with_lease_arg


def test_github_repo_from_slug_parses_owner_and_name() -> None:
    repo = GitHubRepo.from_slug("octocat/hello-world")

    assert repo.owner == "octocat"
    assert repo.name == "hello-world"
    assert repo.slug == "octocat/hello-world"


@pytest.mark.parametrize("slug", ["", "octocat", "octocat/hello/world"])
def test_github_repo_from_slug_rejects_invalid_format(slug: str) -> None:
    with pytest.raises(ValueError):
        GitHubRepo.from_slug(slug)


def test_dvc_tracked_paths_match_directory_sidecars() -> None:
    tracked = dvc_tracked_paths("assets/rag_data/arxiv")

    assert tracked.dataset_rel_path == Path("assets/rag_data/arxiv")
    assert tracked.pointer_rel_path == Path("assets/rag_data/arxiv.dvc")
    assert tracked.gitignore_rel_path == Path("assets/rag_data/.gitignore")


def test_build_github_remote_url_escapes_token() -> None:
    repo = GitHubRepo.from_slug("octocat/hello-world")

    url = build_github_remote_url(repo, "tok:en/with?chars")

    assert (
        url == "https://x-access-token:tok%3Aen%2Fwith%3Fchars@github.com/octocat/hello-world.git"
    )


def test_stage_dataset_path_replaces_existing_dataset_dir(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    source_dir = tmp_path / "shared" / "arxiv"
    destination = clone_root / "assets" / "rag_data" / "arxiv"

    source_dir.mkdir(parents=True)
    (source_dir / "payload.json").write_text("[]", encoding="utf-8")
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("old", encoding="utf-8")

    staged_path = stage_dataset_path(
        clone_root=clone_root,
        dataset_rel_path="assets/rag_data/arxiv",
        source_path=source_dir,
    )

    assert staged_path.is_dir()
    assert not staged_path.is_symlink()
    assert (staged_path / "payload.json").read_text(encoding="utf-8") == "[]"


def test_run_redacts_github_token_in_command_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="fatal: authentication failed",
        )

    monkeypatch.setattr(airflow_git_sync.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"\*\*\*") as exc_info:
        run_command(
            [
                "git",
                "clone",
                "https://x-access-token:secret-token@github.com/octocat/hello-world.git",
            ],
            cwd=tmp_path,
        )

    assert "secret-token" not in str(exc_info.value)


def test_build_force_with_lease_arg_uses_remote_branch_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        airflow_git_sync,
        "_remote_branch_head",
        lambda *, clone_dir, branch: "abc123def456",
    )

    assert build_force_with_lease_arg(clone_dir=tmp_path, branch="data-sync/develop") == (
        "--force-with-lease=refs/heads/data-sync/develop:abc123def456"
    )


def test_build_force_with_lease_arg_requires_branch_absence_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        airflow_git_sync,
        "_remote_branch_head",
        lambda *, clone_dir, branch: None,
    )

    assert build_force_with_lease_arg(clone_dir=tmp_path, branch="data-sync/develop") == (
        "--force-with-lease=refs/heads/data-sync/develop:"
    )


def test_remote_branch_head_parses_ls_remote_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="abc123def456\trefs/heads/data-sync/develop\n",
            stderr="",
        )

    monkeypatch.setattr(airflow_git_sync.subprocess, "run", fake_run)

    assert airflow_git_sync._remote_branch_head(clone_dir=tmp_path, branch="data-sync/develop") == (
        "abc123def456"
    )


def test_remote_branch_head_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(airflow_git_sync.subprocess, "run", fake_run)

    assert (
        airflow_git_sync._remote_branch_head(clone_dir=tmp_path, branch="data-sync/develop") is None
    )
