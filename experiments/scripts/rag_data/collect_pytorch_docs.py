"""Collect PyTorch documentation for code RAG baseline.

Scrapes a curated subset of PyTorch documentation pages.
For baseline: ~50 core API pages (nn.Module, Tensor operations, etc.)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def scrape_pytorch_doc_page(url: str) -> dict:
    """Scrape a single PyTorch documentation page.

    Args:
        url: URL of the documentation page

    Returns:
        Dict with page content and metadata
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Extract title
    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else "Untitled"

    # Extract main content
    content_div = soup.find("div", {"role": "main"}) or soup.find("article")
    if content_div:
        # Remove navigation and other non-content elements
        for tag in content_div.find_all(["nav", "footer", "script", "style"]):
            tag.decompose()
        content = content_div.get_text(separator="\n", strip=True)
    else:
        content = ""

    # Extract code examples
    code_blocks = soup.find_all("code") or soup.find_all("pre")
    code_examples = [block.get_text(strip=True) for block in code_blocks[:10]]  # Limit to 10

    return {
        "url": url,
        "title": title_text,
        "content": content,
        "code_examples": code_examples,
        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def collect_pytorch_docs(
    base_url: str,
    page_list: list[str],
    output_dir: Path,
) -> list[dict]:
    """Collect PyTorch documentation pages.

    Args:
        base_url: Base URL of PyTorch docs
        page_list: List of relative URLs to scrape
        output_dir: Output directory for saved docs

    Returns:
        List of scraped page dicts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    for i, page_path in enumerate(page_list, 1):
        url = urljoin(base_url, page_path)
        print(f"[{i}/{len(page_list)}] Scraping: {url}")

        try:
            page_data = scrape_pytorch_doc_page(url)
            pages.append(page_data)
            time.sleep(1)  # Be nice to the server
        except Exception as e:
            print(f"  Error scraping {url}: {e}")
            continue

    # Save to JSON
    output_file = output_dir / "pytorch_docs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"\nScraped {len(pages)} pages")
    print(f"Saved to: {output_file}")

    return pages


def get_baseline_pages() -> list[str]:
    """Get list of core PyTorch doc pages for baseline.

    Returns:
        List of relative URLs for essential PyTorch docs
    """
    return [
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


def main():
    parser = argparse.ArgumentParser(description="Collect PyTorch documentation for RAG")
    parser.add_argument(
        "--base-url",
        default="https://pytorch.org/docs/stable/",
        help="Base URL of PyTorch documentation",
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        help="Specific pages to scrape (uses baseline list if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/rag_data/pytorch_docs"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Use provided pages or baseline
    page_list = args.pages if args.pages else get_baseline_pages()

    print(f"Collecting {len(page_list)} PyTorch documentation pages...")
    pages = collect_pytorch_docs(
        base_url=args.base_url,
        page_list=page_list,
        output_dir=args.output_dir,
    )

    print("\nSummary:")
    print(f"  Successfully scraped: {len(pages)} pages")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
