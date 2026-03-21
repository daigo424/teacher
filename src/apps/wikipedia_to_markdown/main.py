from __future__ import annotations

import argparse
import re
from pathlib import Path

import html2text
import requests
from bs4 import BeautifulSoup, Tag

CUT_SECTION_TITLES = {
    "footnotes",
    "references",
    "bibliography",
    "external links",
    "see also",
    "notes",
    "further reading",
    "脚注",
    "出典",
    "参考文献",
    "外部リンク",
    "関連項目",
    "注釈",
}

NOISE_CLASSES = {
    "mw-editsection",
    "reference",
    "reflist",
    "navbox",
    "vertical-navbox",
    "metadata",
    "mbox-small",
    "infobox",
    "sidebar",
    "thumb",
    "tright",
    "tleft",
    "toc",
    "authority-control",
    "mw-jump-link",
    "hatnote",
    "shortdescription",
    "sistersitebox",
    "plainlinks",
    "portalbox",
    "succession-box",
    "noprint",
}

NOISE_TAGS = {
    "table",
    "figure",
    "figcaption",
    "audio",
    "video",
    "style",
    "script",
    "noscript",
}


def scrape_wikipedia_to_markdown(url: str) -> tuple[str, str]:
    """
    Scrape a Wikipedia page and convert it into RAG-friendly Markdown.

    Returns:
        tuple[str, str]: (page_title, markdown)
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        title_tag = soup.find("h1", id="firstHeading")
        page_title = title_tag.get_text(strip=True) if title_tag else "Wikipedia"

        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            print("Error: Main content area not found.")
            return "", ""

        cleaned_html = preprocess_wikipedia_html(soup, content_div)

        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_emphasis = False
        h.ignore_tables = True
        h.unicode_snob = True

        markdown = h.handle(str(cleaned_html))
        markdown = postprocess_markdown(markdown)

        final_markdown = f"# {page_title}\n\n{markdown}".strip() + "\n"
        return page_title, final_markdown

    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        return "", ""
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return "", ""


def preprocess_wikipedia_html(soup: BeautifulSoup, content_div: Tag) -> Tag:
    """
    Remove HTML elements that are likely to become noise in RAG.
    """
    content_div = BeautifulSoup(str(content_div), "html.parser").find("div")
    if content_div is None:
        raise ValueError("Failed to clone content_div.")

    # 1. Remove obvious noise tags entirely
    for tag_name in NOISE_TAGS:
        for tag in content_div.find_all(tag_name):
            tag.decompose()

    # 2. Remove elements by noisy CSS classes
    for class_name in NOISE_CLASSES:
        for tag in content_div.find_all(class_=lambda c: c and class_name in c.split()):
            tag.decompose()

    # 3. Remove edit links/buttons
    for tag in content_div.find_all(class_="mw-editsection"):
        tag.decompose()

    # 4. Remove superscript references like [1], [2]
    for sup in content_div.find_all("sup", class_="reference"):
        sup.decompose()

    # 5. Remove "citation needed" / "要出典" like inline notices
    for tag in content_div.find_all(class_=lambda c: c and ("citation-needed" in c.split() or "noprint" in c.split())):
        tag.decompose()

    # 6. Convert <dt> into headings
    for dt_tag in content_div.find_all("dt"):
        text = dt_tag.get_text(" ", strip=True)
        if not text:
            dt_tag.decompose()
            continue

        h4_tag = soup.new_tag("h4")
        h4_tag.string = text
        dt_tag.replace_with(h4_tag)

    # 7. Replace links with their text only
    for a_tag in content_div.find_all("a"):
        text = a_tag.get_text(" ", strip=True)
        a_tag.replace_with(text)

    # 8. Remove spans commonly used for coordinates, IPA, helper UI, etc.
    for span in content_div.find_all("span"):
        classes = set(span.get("class", []))
        if classes & {
            "mw-cite-backlink",
            "geo-inline-hidden",
            "geo-default",
            "latitude",
            "longitude",
            "IPA",
            "nowraplinks",
        }:
            span.decompose()

    # 9. Cut everything after noisy end sections
    cut_after_unwanted_sections(content_div)

    return content_div


def cut_after_unwanted_sections(content_div: Tag) -> None:
    """
    Remove all siblings starting from sections like References / External links.
    """
    for header in content_div.find_all(["h2", "h3"]):
        headline = header.get_text(" ", strip=True).lower()
        headline = headline.replace("[編集]", "").replace("[edit]", "").strip()

        if headline in CUT_SECTION_TITLES:
            current = header
            while current:
                next_node = current.next_sibling
                if isinstance(current, Tag):
                    current.decompose()
                current = next_node
            break


def postprocess_markdown(text: str) -> str:
    """
    Clean Markdown after html2text conversion.
    """
    # Remove leftover citation markers
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[要出典\]", "", text)

    # Remove footnote-like numeric references: [1], [12]
    text = re.sub(r"\[\d+\]", "", text)

    # Remove empty markdown links that may remain
    text = re.sub(r"\[\]\([^)]+\)", "", text)

    # Remove repeated punctuation spacing
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are only separators or whitespace
    text = re.sub(r"(?m)^[ \t]*[-*_]{3,}[ \t]*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def sanitize_filename(title: str) -> str:
    filename = title.replace(" ", "_")
    filename = re.sub(r'[\\/:*?"<>|]', "", filename)
    return filename


if __name__ == "__main__":
    breakpoint()

    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Wikipedia URL")
    args = parser.parse_args()

    page_title, markdown_content = scrape_wikipedia_to_markdown(args.url)

    if markdown_content:
        safe_title = sanitize_filename(page_title)

        output_dir = Path("/files/dataset")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"wikipedia_{safe_title}.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Saved Markdown file to '{output_path}'.")
