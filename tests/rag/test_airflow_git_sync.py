from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

airflow_git_sync = importlib.import_module("shared.airflow_git_sync")
GitHubRepo = airflow_git_sync.GitHubRepo
build_github_remote_url = airflow_git_sync.build_github_remote_url
dvc_tracked_paths = airflow_git_sync.dvc_tracked_paths
replace_with_symlink = airflow_git_sync.replace_with_symlink


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


def test_replace_with_symlink_replaces_existing_dataset_dir(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    source_dir = tmp_path / "shared" / "arxiv"
    destination = clone_root / "assets" / "rag_data" / "arxiv"

    source_dir.mkdir(parents=True)
    (source_dir / "payload.json").write_text("[]", encoding="utf-8")
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("old", encoding="utf-8")

    symlink_path = replace_with_symlink(
        clone_root=clone_root,
        dataset_rel_path="assets/rag_data/arxiv",
        source_dir=source_dir,
    )

    assert symlink_path.is_symlink()
    assert symlink_path.resolve() == source_dir.resolve()
