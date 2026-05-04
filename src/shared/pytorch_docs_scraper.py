"""Shared helpers for scraping PyTorch documentation pages."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

_META_REFRESH_RE = re.compile(r"url\s*=\s*(.+)", re.IGNORECASE)


def _extract_html_redirect_target(soup: BeautifulSoup, *, current_url: str) -> str | None:
    """Return the next URL when a page is an HTML redirect shell."""
    refresh_tag = soup.find(
        "meta",
        attrs={"http-equiv": lambda value: isinstance(value, str) and value.lower() == "refresh"},
    )
    if refresh_tag is not None:
        content = refresh_tag.get("content")
        if isinstance(content, str):
            match = _META_REFRESH_RE.search(content)
            if match is not None:
                return urljoin(current_url, match.group(1).strip(" \"'"))

    page_title = soup.title.get_text(strip=True) if soup.title is not None else ""
    if "redirecting" not in page_title.lower():
        return None

    canonical_tag = soup.find("link", rel="canonical")
    href = canonical_tag.get("href") if canonical_tag is not None else None
    if isinstance(href, str) and href.strip():
        return urljoin(current_url, href.strip())

    continue_link = soup.find("a", href=True)
    href = continue_link.get("href") if continue_link is not None else None
    if isinstance(href, str) and href.strip():
        return urljoin(current_url, href.strip())

    return None


def _looks_like_not_found_page(soup: BeautifulSoup) -> bool:
    """Detect common placeholder pages that should not be indexed."""
    page_title = soup.title.get_text(" ", strip=True).lower() if soup.title is not None else ""
    heading = soup.find("h1")
    heading_text = heading.get_text(" ", strip=True).lower() if heading is not None else ""
    preview = soup.get_text(" ", strip=True).lower()[:500]
    return "page not found" in page_title or heading_text == "404" or "file not found" in preview


def _fetch_resolved_soup(
    url: str,
    *,
    session: requests.Session,
    timeout: float,
    max_html_redirects: int,
) -> tuple[requests.Response, BeautifulSoup]:
    """Fetch a page and follow HTML redirect shells to the real docs page."""
    current_url = url
    visited: set[str] = set()
    response: requests.Response | None = None
    soup: BeautifulSoup | None = None

    for _ in range(max_html_redirects + 1):
        response = session.get(current_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        redirect_target = _extract_html_redirect_target(soup, current_url=response.url)
        if redirect_target is None:
            return response, soup
        if redirect_target in visited or redirect_target == response.url:
            return response, soup

        visited.add(response.url)
        current_url = redirect_target

    assert response is not None
    assert soup is not None
    return response, soup


def scrape_pytorch_doc_page(
    url: str,
    *,
    max_code_examples: int,
    timeout: float = 30,
    max_html_redirects: int = 3,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Scrape one PyTorch docs page, skipping placeholder or empty pages."""
    own_session = session is None
    active_session = session or requests.Session()

    try:
        response, soup = _fetch_resolved_soup(
            url,
            session=active_session,
            timeout=timeout,
            max_html_redirects=max_html_redirects,
        )
    finally:
        if own_session:
            active_session.close()

    if _looks_like_not_found_page(soup):
        return None, f"resolved page looks like a 404 or placeholder ({response.url})"

    title_tag = soup.find("h1")
    title_text = title_tag.get_text(strip=True) if title_tag is not None else "Untitled"

    content_root = soup.find(attrs={"role": "main"}) or soup.find("article")
    if content_root is None:
        return None, f"missing docs content container ({response.url})"

    for tag in content_root.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()

    content = content_root.get_text(separator="\n", strip=True)
    if title_text in {"Untitled", "404"}:
        return None, f"invalid extracted title {title_text!r} ({response.url})"
    if not content:
        return None, f"empty extracted content ({response.url})"

    code_blocks = content_root.find_all("code") or content_root.find_all("pre")
    code_examples = [block.get_text(strip=True) for block in code_blocks[:max_code_examples]]

    return (
        {
            "url": url,
            "title": title_text,
            "content": content,
            "code_examples": code_examples,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        None,
    )
