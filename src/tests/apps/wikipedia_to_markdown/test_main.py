from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from apps.wikipedia_to_markdown.main import (
    cut_after_unwanted_sections,
    postprocess_markdown,
    preprocess_wikipedia_html,
    sanitize_filename,
    scrape_wikipedia_to_markdown,
)


def test_preprocess_wikipedia_html_removes_noise_and_converts_elements():
    soup = BeautifulSoup(
        """
        <div class="mw-parser-output">
            <p>Intro <a href="/wiki/A">Alpha</a><sup class="reference">[1]</sup></p>
            <div class="thumb">remove me</div>
            <dt>Glossary</dt>
            <p><span class="IPA">ipa</span>Body</p>
            <h2>References</h2>
            <p>cut me</p>
        </div>
        """,
        "html.parser",
    )

    cleaned = preprocess_wikipedia_html(soup, soup.find("div", class_="mw-parser-output"))
    cleaned_text = cleaned.get_text(" ", strip=True)

    assert "remove me" not in cleaned_text
    assert "[1]" not in cleaned_text
    assert "cut me" not in cleaned_text
    assert cleaned.find("h4").get_text(strip=True) == "Glossary"
    assert "Alpha" in cleaned_text


def test_cut_after_unwanted_sections_removes_following_nodes():
    soup = BeautifulSoup(
        """
        <div class="mw-parser-output">
            <h2>History</h2><p>keep</p>
            <h2>External links</h2><p>drop</p><p>drop too</p>
        </div>
        """,
        "html.parser",
    )
    content = soup.find("div", class_="mw-parser-output")

    cut_after_unwanted_sections(content)

    assert "keep" in content.get_text(" ", strip=True)
    assert "drop" not in content.get_text(" ", strip=True)


def test_postprocess_markdown_cleans_noise():
    cleaned = postprocess_markdown("Line [1]\n\n\n[citation needed]\n\n---\n\nText  [](/x)")

    assert cleaned == "Line \n\nText"


def test_sanitize_filename_removes_unsafe_characters():
    assert sanitize_filename('AI / ML: "Guide"?') == "AI__ML_Guide"


def test_scrape_wikipedia_to_markdown_returns_markdown(monkeypatch):
    html = """
    <html>
      <h1 id="firstHeading">Neural network</h1>
      <div class="mw-parser-output">
        <p>A model for learning.</p>
      </div>
    </html>
    """

    class FakeResponse:
        content = html.encode("utf-8")

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "apps.wikipedia_to_markdown.main.requests.get",
        lambda url, headers, timeout: FakeResponse(),
    )

    title, markdown = scrape_wikipedia_to_markdown("https://example.com")

    assert title == "Neural network"
    assert markdown.startswith("# Neural network")
    assert "A model for learning." in markdown


def test_scrape_wikipedia_to_markdown_returns_empty_on_request_error(monkeypatch, capsys):
    def raise_error(url, headers, timeout):
        raise RequestException("boom")

    monkeypatch.setattr("apps.wikipedia_to_markdown.main.requests.get", raise_error)

    assert scrape_wikipedia_to_markdown("https://example.com") == ("", "")
    assert "HTTP request error" in capsys.readouterr().out
