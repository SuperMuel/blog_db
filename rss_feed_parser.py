import logging
import re
from datetime import datetime
from typing import Iterable

from markdownify import markdownify as md
from pydantic import BaseModel

from models import HttpUrl

logger = logging.getLogger(__name__)


class RssArticle(BaseModel):
    title: str
    url: str
    published: datetime | None
    markdown_content: str


def remove_extra_newlines(markdown: str) -> str:
    """
    Remove extra consecutive newline characters from a markdown string.

    Args:
        markdown (str): The markdown string to process.

    Returns:
        str: The processed markdown string with no more than one consecutive newline character.
    """
    return re.sub(r"\n{2,}", "\n", markdown)


def entry_to_markdown(entry) -> None | str:
    html = None
    try:
        html = entry["content"][0]["value"]
    except Exception:
        html = entry.get("description", entry.get("summary", ""))

    if not html:
        logger.warning(f"Failed to extract content from entry: {entry}")
        return None

    markdown = md(html)
    markdown = remove_extra_newlines(markdown)

    return markdown


def entry_to_published_date(entry) -> datetime | None:
    try:
        return datetime(*entry.published_parsed[:6])
    except Exception:
        logger.warning(f"Failed to extract published date from entry: {entry}")
        return None


class RSSFeedParser:
    def __init__(self, url: str | HttpUrl):
        self.url = str(url)
        self.feed = None

    def parse(self):
        import feedparser

        self.feed = feedparser.parse(self.url)

    def get_number_of_entries(self):
        if self.feed is None:
            self.parse()
        return len(self.feed.entries)  # type: ignore

    def get_entries(self) -> Iterable[RssArticle]:
        if self.feed is None:
            self.parse()
        for entry in self.feed.entries:  # type: ignore
            markdown_content = entry_to_markdown(entry)

            if markdown_content is None:
                continue

            yield RssArticle(
                title=entry.title,
                url=entry.link,
                published=entry_to_published_date(entry),
                markdown_content=markdown_content,
            )
