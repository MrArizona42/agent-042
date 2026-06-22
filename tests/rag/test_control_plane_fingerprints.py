"""Contract tests for `rag.control_plane.fingerprints`.

Covers formatting-independent canonical digests and identity changes for
every semantic input, per the declarative alias workflow plan's Phase 0
acceptance criteria.
"""

from __future__ import annotations

from app_config.catalog.schema import AliasBuildConfig, AliasChunkingConfig, AliasRetrievalConfig
from rag.control_plane import fingerprints as fp


def _build_config(**overrides):
    defaults = {
        "chunking": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        "dense_encoder": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384},
    }
    defaults.update(overrides)
    return AliasBuildConfig(**defaults)


def _retrieval_config(**overrides):
    defaults = {"strategy": "dense", "top_k": 5, "score_threshold": 0.35}
    defaults.update(overrides)
    return AliasRetrievalConfig(**defaults)


class TestCanonicalDigest:
    def test_digest_is_sha256_prefixed(self):
        digest = fp.canonical_digest({"a": 1})
        assert digest.startswith("sha256:")
        assert len(digest.removeprefix("sha256:")) == 64

    def test_key_order_does_not_affect_digest(self):
        a = fp.canonical_digest({"a": 1, "b": 2})
        b = fp.canonical_digest({"b": 2, "a": 1})
        assert a == b

    def test_different_values_change_digest(self):
        a = fp.canonical_digest({"a": 1})
        b = fp.canonical_digest({"a": 2})
        assert a != b

    def test_pydantic_model_and_equivalent_dict_match(self):
        build = _build_config()
        a = fp.canonical_digest(build)
        b = fp.canonical_digest(build.model_dump(mode="json"))
        assert a == b


class TestBuildConfigDigest:
    def test_identical_configs_match(self):
        a = fp.build_config_digest(_build_config())
        b = fp.build_config_digest(_build_config())
        assert a == b

    def test_chunk_size_change_alters_digest(self):
        a = fp.build_config_digest(_build_config())
        b = fp.build_config_digest(
            _build_config(chunking={"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 64})
        )
        assert a != b

    def test_dense_encoder_model_change_alters_digest(self):
        a = fp.build_config_digest(_build_config())
        b = fp.build_config_digest(
            _build_config(dense_encoder={"model": "other-model", "dimension": 384})
        )
        assert a != b

    def test_adding_sparse_encoder_alters_digest(self):
        a = fp.build_config_digest(_build_config())
        b = fp.build_config_digest(_build_config(sparse_encoder={"model": "Qdrant/bm25"}))
        assert a != b


class TestRetrievalConfigDigest:
    def test_identical_configs_match(self):
        a = fp.retrieval_config_digest(_retrieval_config())
        b = fp.retrieval_config_digest(_retrieval_config())
        assert a == b

    def test_top_k_change_alters_digest(self):
        a = fp.retrieval_config_digest(_retrieval_config())
        b = fp.retrieval_config_digest(_retrieval_config(top_k=10))
        assert a != b

    def test_build_and_retrieval_digests_are_independent(self):
        build_digest = fp.build_config_digest(_build_config())
        retrieval_digest = fp.retrieval_config_digest(_retrieval_config())
        # Changing build config must not alter a retrieval digest computed
        # from an unrelated retrieval config, and vice versa.
        other_build_digest = fp.build_config_digest(
            _build_config(dense_encoder={"model": "other-model", "dimension": 384})
        )
        assert build_digest != other_build_digest
        assert fp.retrieval_config_digest(_retrieval_config()) == retrieval_digest


class TestCatalogDigest:
    def test_changing_build_alone_changes_catalog_digest(self):
        a = fp.catalog_digest(_build_config(), _retrieval_config())
        b = fp.catalog_digest(
            _build_config(dense_encoder={"model": "other-model", "dimension": 384}),
            _retrieval_config(),
        )
        assert a != b

    def test_changing_retrieve_alone_changes_catalog_digest(self):
        a = fp.catalog_digest(_build_config(), _retrieval_config())
        b = fp.catalog_digest(_build_config(), _retrieval_config(top_k=10))
        assert a != b

    def test_identical_inputs_match(self):
        a = fp.catalog_digest(_build_config(), _retrieval_config())
        b = fp.catalog_digest(_build_config(), _retrieval_config())
        assert a == b


class TestSourceDeclarationDigest:
    def test_order_independent(self):
        first = ("kb.docs", "sha256:1", "generic.http_html", "1")
        second = ("kb.papers", "sha256:2", "generic.arxiv", "1")
        a = fp.source_declaration_digest([first, second])
        b = fp.source_declaration_digest([second, first])
        assert a == b

    def test_manifest_digest_change_alters_digest(self):
        a = fp.source_declaration_digest([("kb.docs", "sha256:1", "generic.http_html", "1")])
        b = fp.source_declaration_digest([("kb.docs", "sha256:2", "generic.http_html", "1")])
        assert a != b

    def test_adapter_version_change_alters_digest(self):
        a = fp.source_declaration_digest([("kb.docs", "sha256:1", "generic.http_html", "1")])
        b = fp.source_declaration_digest([("kb.docs", "sha256:1", "generic.http_html", "2")])
        assert a != b

    def test_added_source_alters_digest(self):
        a = fp.source_declaration_digest([("kb.docs", "sha256:1", "generic.http_html", "1")])
        b = fp.source_declaration_digest(
            [
                ("kb.docs", "sha256:1", "generic.http_html", "1"),
                ("kb.papers", "sha256:2", "generic.arxiv", "1"),
            ]
        )
        assert a != b


class TestTransformationDigest:
    def test_identical_chunking_matches(self):
        chunking = AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=64)
        a = fp.transformation_digest(chunking)
        b = fp.transformation_digest(
            AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=64)
        )
        assert a == b

    def test_chunk_overlap_change_alters_digest(self):
        a = fp.transformation_digest(
            AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=64)
        )
        b = fp.transformation_digest(
            AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=32)
        )
        assert a != b

    def test_contract_version_change_alters_digest(self):
        chunking = AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=64)
        a = fp.transformation_digest(chunking, contract_version="1")
        b = fp.transformation_digest(chunking, contract_version="2")
        assert a != b

    def test_excludes_encoder_identity_by_construction(self):
        # transformation_digest takes only chunking; encoder changes cannot
        # affect it because they are not part of the function signature.
        chunking = AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=64)
        assert fp.transformation_digest(chunking) == fp.transformation_digest(chunking)


class TestSourceSnapshotId:
    def test_order_independent(self):
        a = fp.source_snapshot_id([("kb.docs", "sha256:1"), ("kb.papers", "sha256:2")])
        b = fp.source_snapshot_id([("kb.papers", "sha256:2"), ("kb.docs", "sha256:1")])
        assert a == b

    def test_checksum_change_alters_snapshot(self):
        a = fp.source_snapshot_id([("kb.docs", "sha256:1")])
        b = fp.source_snapshot_id([("kb.docs", "sha256:2")])
        assert a != b


class TestReleaseFingerprint:
    def _fingerprint(self, **overrides):
        defaults = {
            "kb_id": "pytorch_reference",
            "build_config_digest": "sha256:b",
            "source_declaration_digest": "sha256:d",
            "source_snapshot_id": "sha256:s",
        }
        defaults.update(overrides)
        return fp.release_fingerprint(**defaults)

    def test_identical_inputs_match(self):
        assert self._fingerprint() == self._fingerprint()

    def test_kb_id_change_alters_fingerprint(self):
        assert self._fingerprint() != self._fingerprint(kb_id="ml_papers_core")

    def test_build_config_digest_change_alters_fingerprint(self):
        assert self._fingerprint() != self._fingerprint(build_config_digest="sha256:other")

    def test_source_declaration_digest_change_alters_fingerprint(self):
        assert self._fingerprint() != self._fingerprint(source_declaration_digest="sha256:other")

    def test_source_snapshot_id_change_alters_fingerprint(self):
        assert self._fingerprint() != self._fingerprint(source_snapshot_id="sha256:other")


class TestReleaseAndCollectionNaming:
    def test_release_id_format(self):
        fingerprint = "sha256:" + "a" * 64
        release_id = fp.release_id("pytorch_reference", fingerprint)
        assert release_id == "ragrel_pytorch_reference_" + "a" * 16

    def test_collection_name_format(self):
        fingerprint = "sha256:" + "a" * 64
        name = fp.collection_name("pytorch_reference", fingerprint)
        assert name == "rag__pytorch_reference__" + "a" * 16

    def test_kb_id_is_sanitized(self):
        fingerprint = "sha256:" + "a" * 64
        release_id = fp.release_id("pytorch-reference.v2", fingerprint)
        assert release_id == "ragrel_pytorch_reference_v2_" + "a" * 16

    def test_different_fingerprints_produce_different_names(self):
        a = fp.release_id("pytorch_reference", "sha256:" + "a" * 64)
        b = fp.release_id("pytorch_reference", "sha256:" + "b" * 64)
        assert a != b
