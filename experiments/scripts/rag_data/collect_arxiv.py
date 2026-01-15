"""Collect ArXiv papers for chat RAG baseline.

Downloads a small set of recent ML/DL papers from arXiv.
For baseline: ~100 papers from cs.LG and cs.AI categories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arxiv


def download_arxiv_papers(
    categories: list[str],
    max_results: int,
    output_dir: Path,
) -> list[dict]:
    """Download papers from arXiv and save metadata + abstracts.

    Args:
        categories: List of arXiv categories (e.g., ['cs.LG', 'cs.AI'])
        max_results: Maximum number of papers to download
        output_dir: Directory to save papers

    Returns:
        List of paper metadata dicts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build search query
    query = " OR ".join([f"cat:{cat}" for cat in categories])
    print(f"Searching arXiv with query: {query}")
    print(f"Maximum results: {max_results}")

    # Search arXiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for i, result in enumerate(client.results(search), 1):
        paper = {
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat(),
            "categories": result.categories,
            "primary_category": result.primary_category,
            "pdf_url": result.pdf_url,
        }
        papers.append(paper)

        if i % 10 == 0:
            print(f"Downloaded metadata for {i} papers...")

    # Save all papers to JSON
    output_file = output_dir / "arxiv_papers.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\nDownloaded {len(papers)} papers")
    print(f"Saved to: {output_file}")

    return papers


def main():
    parser = argparse.ArgumentParser(description="Download ArXiv papers for RAG")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.LG", "cs.AI"],
        help="ArXiv categories to download from",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of papers to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/rag_data/arxiv"),
        help="Output directory for downloaded papers",
    )

    args = parser.parse_args()

    papers = download_arxiv_papers(
        categories=args.categories,
        max_results=args.max_results,
        output_dir=args.output_dir,
    )

    print("\nSummary:")
    print(f"  Total papers: {len(papers)}")
    print(f"  Categories: {', '.join(args.categories)}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
