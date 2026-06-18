from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
scrape_pytorch_doc_page = importlib.import_module(
    "rag.adapters.pytorch_docs"
).scrape_pytorch_doc_page


@dataclass
class FakeResponse:
    url: str
    text: str
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code} for {self.url}")


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        del timeout
        self.calls.append(url)
        return self._responses[url]


def test_scrape_pytorch_doc_page_follows_html_redirect_shell() -> None:
    original_url = "https://pytorch.org/docs/stable/generated/torch.nn.Module.html"
    stable_url = "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html"
    versioned_url = "https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html"
    redirect_shell = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Redirecting&hellip;</title>
        <meta http-equiv="refresh" content="0; url=../../2.11/generated/torch.nn.Module.html">
        <link rel="canonical" href="../../2.11/generated/torch.nn.Module.html">
      </head>
      <body>
        <a href="../../2.11/generated/torch.nn.Module.html">Continue to docs</a>
      </body>
    </html>
    """
    docs_html = """
    <html>
      <head><title>Module - PyTorch 2.11 documentation</title></head>
      <body>
        <article>
          <h1>Module</h1>
          <p>Base class for all neural network modules.</p>
          <pre>class MyModule(nn.Module): ...</pre>
        </article>
      </body>
    </html>
    """
    session = FakeSession(
        {
            original_url: FakeResponse(url=stable_url, text=redirect_shell),
            versioned_url: FakeResponse(url=versioned_url, text=docs_html),
        }
    )

    page, skip_reason = scrape_pytorch_doc_page(
        original_url,
        max_code_examples=10,
        session=session,
    )

    assert skip_reason is None
    assert page is not None
    assert page["url"] == original_url
    assert page["title"] == "Module"
    assert "Base class for all neural network modules." in page["content"]
    assert page["code_examples"] == ["class MyModule(nn.Module): ..."]
    assert session.calls == [original_url, versioned_url]


def test_scrape_pytorch_doc_page_skips_placeholder_page() -> None:
    url = "https://pytorch.org/docs/stable/generated/torch.Tensor.html"
    placeholder_html = """
    <!DOCTYPE html>
    <html>
      <head><title>Page not found · GitHub Pages</title></head>
      <body>
        <div class="container">
          <h1>404</h1>
          <p><strong>File not found</strong></p>
        </div>
      </body>
    </html>
    """
    session = FakeSession({url: FakeResponse(url=url, text=placeholder_html)})

    page, skip_reason = scrape_pytorch_doc_page(
        url,
        max_code_examples=10,
        session=session,
    )

    assert page is None
    assert skip_reason is not None
    assert "404 or placeholder" in skip_reason
