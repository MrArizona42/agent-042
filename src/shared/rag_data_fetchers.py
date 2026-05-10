"""Reusable fetch helpers for RAG source datasets."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

DEFAULT_ARXIV_CATEGORIES: list[str] = [
    "cs.LG",
    "stat.ML",
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "cs.NE",
    "cs.RO",
]
DEFAULT_ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_ARXIV_PAGE_SIZE = 100
DEFAULT_ARXIV_DELAY_SECONDS = 4.0
DEFAULT_ARXIV_MAX_RETRIES = 5
DEFAULT_ARXIV_REQUEST_TIMEOUT = 60
DEFAULT_ARXIV_OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
DEFAULT_ARXIV_OAI_METADATA_PREFIX = "arXiv"
DEFAULT_ARXIV_OAI_SET_SPECS: list[str] = [
    "cs:cs:AI",
    "cs:cs:CL",
    "cs:cs:CV",
    "cs:cs:IR",
    "cs:cs:LG",
    "cs:cs:NE",
    "cs:cs:RO",
    "stat:stat:ML",
]
DEFAULT_ARXIV_OAI_DELAY_SECONDS = 2.0
ARXIV_XML_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}
ARXIV_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}

DEFAULT_ARXIV_S3_BUCKET = "s3://arxiv"
DEFAULT_ARXIV_S3_PDF_MANIFEST_URL = f"{DEFAULT_ARXIV_S3_BUCKET}/pdf/arXiv_pdf_manifest.xml"
DEFAULT_ARXIV_S3_SOURCE_MANIFEST_URL = f"{DEFAULT_ARXIV_S3_BUCKET}/src/arXiv_src_manifest.xml"
DEFAULT_ARXIV_S3_BULK_DOCS_URL = "https://info.arxiv.org/help/bulk_data_s3.html"

DEFAULT_PYTORCH_BASE_URL = "https://pytorch.org/docs/stable/"
DEFAULT_PYTORCH_SCRAPE_DELAY_SECONDS = 1.0
DEFAULT_PYTORCH_MAX_CODE_EXAMPLES = 1000
DEFAULT_PYTORCH_PAGES: list[str] = [
    "generated/torch.nn.Module.html",
    "generated/torch.Tensor.html",
    "generated/torch.nn.Linear.html",
    "generated/torch.nn.Conv2d.html",
    "generated/torch.nn.functional.relu.html",
    "generated/torch.optim.Adam.html",
    "generated/torch.optim.SGD.html",
    "generated/torch.nn.CrossEntropyLoss.html",
    "generated/torch.nn.MSELoss.html",
    "generated/torch.autograd.backward.html",
    "tensors.html",
    "autograd.html",
    "nn.html",
    "optim.html",
    "torch.html",
]


def build_arxiv_query(categories: Sequence[str]) -> str:
    """Build an arXiv query that matches any of the requested categories."""
    return "(" + " OR ".join(f"cat:{category}" for category in categories) + ")"


def _parse_retry_after(retry_after: str | None) -> float | None:
    if retry_after is None:
        return None

    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        return None


def _fetch_arxiv_feed(
    search_query: str,
    *,
    start: int,
    max_results: int,
    request_timeout: int,
    max_retries: int,
    base_delay_seconds: float,
    user_agent: str,
    api_url: str,
) -> ET.Element:
    params = urlencode(
        {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = Request(
        f"{api_url}?{params}",
        headers={"User-Agent": user_agent},
    )

    backoff_seconds = max(base_delay_seconds, 1.0)
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(request, timeout=request_timeout) as response:
                return ET.fromstring(response.read())
        except HTTPError as exc:
            should_retry = exc.code in {429, 500, 502, 503, 504} and attempt < max_retries
            if not should_retry:
                raise

            sleep_seconds = _parse_retry_after(exc.headers.get("Retry-After")) or backoff_seconds
            print(
                f"  HTTP {exc.code} from arXiv; "
                f"sleeping {sleep_seconds:.1f}s before retry {attempt}/{max_retries}"
            )
            time.sleep(sleep_seconds)
            backoff_seconds *= 2
        except URLError as exc:
            if attempt == max_retries:
                raise

            print(
                f"  Network error from arXiv ({exc.reason}); "
                f"sleeping {backoff_seconds:.1f}s before retry {attempt}/{max_retries}"
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    raise RuntimeError("arXiv request retry loop exhausted unexpectedly")


def get_arxiv_total_results(
    categories: Sequence[str],
    *,
    request_timeout: int = DEFAULT_ARXIV_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_ARXIV_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_ARXIV_DELAY_SECONDS,
    user_agent: str = "agent-042-rag-data-fetchers/1.0",
    api_url: str = DEFAULT_ARXIV_API_URL,
) -> int:
    """Return the current number of search matches for the category query."""
    query = build_arxiv_query(categories)
    feed = _fetch_arxiv_feed(
        query,
        start=0,
        max_results=0,
        request_timeout=request_timeout,
        max_retries=max_retries,
        base_delay_seconds=base_delay_seconds,
        user_agent=user_agent,
        api_url=api_url,
    )
    total_results = feed.findtext("opensearch:totalResults", default="0", namespaces=ARXIV_XML_NS)
    return int(total_results)


def _normalize_arxiv_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _parse_arxiv_entry(entry: ET.Element) -> dict[str, Any]:
    entry_id = entry.findtext("atom:id", default="", namespaces=ARXIV_XML_NS)
    primary_category = entry.find("arxiv:primary_category", ARXIV_XML_NS)

    pdf_url = None
    for link in entry.findall("atom:link", ARXIV_XML_NS):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href")
            break

    return {
        "arxiv_id": entry_id.rsplit("/", 1)[-1],
        "title": _normalize_arxiv_text(entry.findtext("atom:title", default="", namespaces=ARXIV_XML_NS)),
        "authors": [
            _normalize_arxiv_text(author.findtext("atom:name", default="", namespaces=ARXIV_XML_NS))
            for author in entry.findall("atom:author", ARXIV_XML_NS)
        ],
        "abstract": _normalize_arxiv_text(
            entry.findtext("atom:summary", default="", namespaces=ARXIV_XML_NS)
        ),
        "published": entry.findtext("atom:published", default="", namespaces=ARXIV_XML_NS),
        "updated": entry.findtext("atom:updated", default="", namespaces=ARXIV_XML_NS),
        "categories": [
            category.attrib["term"]
            for category in entry.findall("atom:category", ARXIV_XML_NS)
            if category.attrib.get("scheme") == ARXIV_XML_NS["arxiv"]
        ],
        "primary_category": primary_category.attrib.get("term") if primary_category is not None else None,
        "pdf_url": pdf_url,
    }


def download_arxiv_papers(
    categories: Sequence[str],
    max_results: int,
    output_dir: str | Path,
    *,
    page_size: int = DEFAULT_ARXIV_PAGE_SIZE,
    delay_seconds: float = DEFAULT_ARXIV_DELAY_SECONDS,
    max_retries: int = DEFAULT_ARXIV_MAX_RETRIES,
    request_timeout: int = DEFAULT_ARXIV_REQUEST_TIMEOUT,
    user_agent: str = "agent-042-rag-data-fetchers/1.0",
    api_url: str = DEFAULT_ARXIV_API_URL,
) -> dict[str, Any]:
    """Download arXiv metadata into the target directory and return a small summary."""
    if max_results < 0:
        raise ValueError("max_results must be >= 0")
    if not 1 <= page_size <= 2000:
        raise ValueError("page_size must be between 1 and 2000")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    query = build_arxiv_query(categories)
    total_available = get_arxiv_total_results(
        categories,
        request_timeout=request_timeout,
        max_retries=max_retries,
        base_delay_seconds=delay_seconds,
        user_agent=user_agent,
        api_url=api_url,
    )
    target_results = min(max_results, total_available)

    print(f"Searching arXiv: {query}")
    print(f"Matched {total_available} papers across {len(categories)} categories.")
    print(
        f"Downloading {target_results} papers "
        f"with page_size={page_size} and delay={delay_seconds:.1f}s between requests."
    )

    papers: list[dict[str, Any]] = []
    for start in range(0, target_results, page_size):
        batch_size = min(page_size, target_results - start)
        feed = _fetch_arxiv_feed(
            query,
            start=start,
            max_results=batch_size,
            request_timeout=request_timeout,
            max_retries=max_retries,
            base_delay_seconds=delay_seconds,
            user_agent=user_agent,
            api_url=api_url,
        )
        entries = feed.findall("atom:entry", ARXIV_XML_NS)
        if not entries:
            print(f"  No entries returned at offset {start}; stopping early.")
            break

        papers.extend(_parse_arxiv_entry(entry) for entry in entries)
        print(f"  fetched {len(papers)}/{target_results} papers")

        if len(entries) < batch_size:
            print("  arXiv returned fewer records than requested; stopping early.")
            break

        if len(papers) < target_results:
            time.sleep(delay_seconds)

    output_file = output_path / "arxiv_papers.json"
    with open(output_file, "w", encoding="utf-8") as file_handle:
        json.dump(papers, file_handle, indent=2, ensure_ascii=False)

    metadata_file = output_path / "arxiv_fetch_metadata.json"
    summary = {
        "query": query,
        "categories": list(categories),
        "total_available": total_available,
        "requested_results": max_results,
        "downloaded_results": len(papers),
        "page_size": page_size,
        "delay_seconds": delay_seconds,
        "output_file": str(output_file),
        "metadata_file": str(metadata_file),
    }
    with open(metadata_file, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, ensure_ascii=False)

    print(f"Downloaded {len(papers)} papers -> {output_file}")
    print(f"Saved fetch metadata -> {metadata_file}")
    return summary


def _fetch_arxiv_oai_page(
    params: dict[str, str],
    *,
    base_url: str,
    request_timeout: int,
    max_retries: int,
    base_delay_seconds: float,
    user_agent: str,
) -> ET.Element:
    request = Request(
        f"{base_url}?{urlencode(params)}",
        headers={"User-Agent": user_agent},
    )

    backoff_seconds = max(base_delay_seconds, 1.0)
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(request, timeout=request_timeout) as response:
                return ET.fromstring(response.read())
        except HTTPError as exc:
            should_retry = exc.code in {429, 500, 502, 503, 504} and attempt < max_retries
            if not should_retry:
                raise

            sleep_seconds = _parse_retry_after(exc.headers.get("Retry-After")) or backoff_seconds
            print(
                f"  HTTP {exc.code} from arXiv OAI-PMH; "
                f"sleeping {sleep_seconds:.1f}s before retry {attempt}/{max_retries}"
            )
            time.sleep(sleep_seconds)
            backoff_seconds *= 2
        except URLError as exc:
            if attempt == max_retries:
                raise

            print(
                f"  Network error from arXiv OAI-PMH ({exc.reason}); "
                f"sleeping {backoff_seconds:.1f}s before retry {attempt}/{max_retries}"
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    raise RuntimeError("arXiv OAI-PMH request retry loop exhausted unexpectedly")


def _parse_arxiv_oai_author(author: ET.Element) -> str:
    forenames = _normalize_arxiv_text(
        author.findtext("arxiv:forenames", default="", namespaces=ARXIV_OAI_NS)
    )
    keyname = _normalize_arxiv_text(
        author.findtext("arxiv:keyname", default="", namespaces=ARXIV_OAI_NS)
    )
    return " ".join(part for part in (forenames, keyname) if part)


def _parse_arxiv_oai_record(record: ET.Element) -> dict[str, Any] | None:
    header = record.find("oai:header", ARXIV_OAI_NS)
    if header is None or header.attrib.get("status") == "deleted":
        return None

    metadata = record.find("oai:metadata", ARXIV_OAI_NS)
    arxiv_record = metadata.find("arxiv:arXiv", ARXIV_OAI_NS) if metadata is not None else None
    if arxiv_record is None:
        return None

    categories_text = _normalize_arxiv_text(
        arxiv_record.findtext("arxiv:categories", default="", namespaces=ARXIV_OAI_NS)
    )
    categories = categories_text.split() if categories_text else []

    return {
        "oai_identifier": header.findtext("oai:identifier", default="", namespaces=ARXIV_OAI_NS),
        "datestamp": header.findtext("oai:datestamp", default="", namespaces=ARXIV_OAI_NS),
        "set_specs": [
            set_spec.text.strip()
            for set_spec in header.findall("oai:setSpec", ARXIV_OAI_NS)
            if set_spec.text and set_spec.text.strip()
        ],
        "arxiv_id": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:id", default="", namespaces=ARXIV_OAI_NS)
        ),
        "created": arxiv_record.findtext("arxiv:created", default="", namespaces=ARXIV_OAI_NS),
        "title": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:title", default="", namespaces=ARXIV_OAI_NS)
        ),
        "authors": [
            author_name
            for author_name in (
                _parse_arxiv_oai_author(author)
                for author in arxiv_record.findall("arxiv:authors/arxiv:author", ARXIV_OAI_NS)
            )
            if author_name
        ],
        "abstract": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:abstract", default="", namespaces=ARXIV_OAI_NS)
        ),
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "comments": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:comments", default="", namespaces=ARXIV_OAI_NS)
        ),
        "journal_ref": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:journal-ref", default="", namespaces=ARXIV_OAI_NS)
        ),
        "doi": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:doi", default="", namespaces=ARXIV_OAI_NS)
        ),
        "license": _normalize_arxiv_text(
            arxiv_record.findtext("arxiv:license", default="", namespaces=ARXIV_OAI_NS)
        ),
    }


def harvest_arxiv_metadata_oai(
    set_specs: Sequence[str],
    output_dir: str | Path,
    *,
    base_url: str = DEFAULT_ARXIV_OAI_BASE_URL,
    metadata_prefix: str = DEFAULT_ARXIV_OAI_METADATA_PREFIX,
    delay_seconds: float = DEFAULT_ARXIV_OAI_DELAY_SECONDS,
    request_timeout: int = DEFAULT_ARXIV_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_ARXIV_MAX_RETRIES,
    max_pages_per_set: int | None = None,
    user_agent: str = "agent-042-rag-data-fetchers-oai/1.0",
) -> dict[str, Any]:
    """Harvest bulk arXiv metadata via the official OAI-PMH endpoint.

    Records are written as JSONL, one file per OAI set, so large harvests do not
    have to fit in memory.
    """
    if not set_specs:
        raise ValueError("set_specs must contain at least one OAI-PMH set")
    if max_pages_per_set is not None and max_pages_per_set <= 0:
        raise ValueError("max_pages_per_set must be greater than zero when provided")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    set_summaries: list[dict[str, Any]] = []
    total_records = 0
    total_deleted_records = 0

    for set_spec in set_specs:
        print(f"Harvesting OAI-PMH set: {set_spec}")
        safe_set_name = set_spec.replace(":", "__")
        output_file = output_path / f"{safe_set_name}.jsonl"

        record_count = 0
        deleted_records = 0
        page_count = 0
        next_resumption_token: str | None = None

        with open(output_file, "w", encoding="utf-8") as file_handle:
            resumption_token: str | None = None
            while True:
                if resumption_token:
                    params = {
                        "verb": "ListRecords",
                        "resumptionToken": resumption_token,
                    }
                else:
                    params = {
                        "verb": "ListRecords",
                        "metadataPrefix": metadata_prefix,
                        "set": set_spec,
                    }

                root = _fetch_arxiv_oai_page(
                    params,
                    base_url=base_url,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    base_delay_seconds=delay_seconds,
                    user_agent=user_agent,
                )
                page_count += 1

                for record in root.findall(".//oai:record", ARXIV_OAI_NS):
                    header = record.find("oai:header", ARXIV_OAI_NS)
                    if header is not None and header.attrib.get("status") == "deleted":
                        deleted_records += 1
                        continue

                    parsed_record = _parse_arxiv_oai_record(record)
                    if parsed_record is None:
                        continue

                    json.dump(parsed_record, file_handle, ensure_ascii=False)
                    file_handle.write("\n")
                    record_count += 1

                token_text = root.findtext(
                    ".//oai:resumptionToken",
                    default="",
                    namespaces=ARXIV_OAI_NS,
                ).strip()
                next_resumption_token = token_text or None

                if max_pages_per_set is not None and page_count >= max_pages_per_set:
                    break
                if not next_resumption_token:
                    break
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

                resumption_token = next_resumption_token

        total_records += record_count
        total_deleted_records += deleted_records

        set_summary = {
            "set_spec": set_spec,
            "page_count": page_count,
            "record_count": record_count,
            "deleted_records": deleted_records,
            "output_file": str(output_file),
            "next_resumption_token": next_resumption_token,
            "stopped_early": bool(next_resumption_token),
        }
        set_summaries.append(set_summary)
        print(
            f"  wrote {record_count} records from {page_count} pages -> {output_file}"
        )

    summary = {
        "base_url": base_url,
        "metadata_prefix": metadata_prefix,
        "set_specs": list(set_specs),
        "total_records": total_records,
        "total_deleted_records": total_deleted_records,
        "output_dir": str(output_path),
        "sets": set_summaries,
        "official_full_text": {
            "docs_url": DEFAULT_ARXIV_S3_BULK_DOCS_URL,
            "pdf_manifest_url": DEFAULT_ARXIV_S3_PDF_MANIFEST_URL,
            "source_manifest_url": DEFAULT_ARXIV_S3_SOURCE_MANIFEST_URL,
        },
    }
    summary_file = output_path / "oai_harvest_summary.json"
    summary["summary_file"] = str(summary_file)

    with open(summary_file, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, ensure_ascii=False)

    print(f"Saved OAI-PMH harvest summary -> {summary_file}")
    return summary


def collect_pytorch_docs(
    base_url: str,
    page_list: Sequence[str],
    output_dir: str | Path,
    *,
    delay_seconds: float = DEFAULT_PYTORCH_SCRAPE_DELAY_SECONDS,
    max_code_examples: int = DEFAULT_PYTORCH_MAX_CODE_EXAMPLES,
) -> dict[str, Any]:
    """Scrape PyTorch docs pages into the target directory and return a small summary."""
    from shared.pytorch_docs_scraper import scrape_pytorch_doc_page

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    skipped_pages: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for index, page_path in enumerate(page_list, 1):
        url = urljoin(base_url, page_path)
        print(f"[{index}/{len(page_list)}] {url}")
        try:
            page, skip_reason = scrape_pytorch_doc_page(
                url,
                max_code_examples=max_code_examples,
            )
            if page is None:
                skipped_pages.append({"url": url, "reason": skip_reason or "unknown"})
                print(f"  Warning: skipped page ({skip_reason})")
            else:
                pages.append(page)
            if index < len(page_list) and delay_seconds > 0:
                time.sleep(delay_seconds)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            print(f"  Warning: {exc}")

    output_file = output_path / "pytorch_docs.json"
    with open(output_file, "w", encoding="utf-8") as file_handle:
        json.dump(pages, file_handle, indent=2, ensure_ascii=False)

    print(f"Scraped {len(pages)} pages -> {output_file}")
    return {
        "base_url": base_url,
        "requested_pages": len(page_list),
        "scraped_pages": len(pages),
        "skipped_pages": skipped_pages,
        "errors": errors,
        "output_file": str(output_file),
    }