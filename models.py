from enum import Enum
from typing import Annotated

from beanie import Document
from pydantic import (
    Field,
    UrlConstraints,
)
from pydantic_core import Url
from pymongo import IndexModel

HttpUrl = Annotated[
    Url,
    UrlConstraints(max_length=2083, allowed_schemes=["http", "https"]),
]


class User(Document):
    username: str = Field(
        ...,
        max_length=30,
        examples=["SuperMuel"],
    )
    api_key: str = Field(
        ...,
        max_length=100,
        min_length=5,
        examples=["secret"],
        repr=False,
        exclude=True,
    )

    class Settings:
        name = "users"
        indexes = [
            IndexModel("username", unique=True),
            IndexModel("api_key"),
        ]


class Article(Document):
    title: str = Field(
        ...,
        max_length=300,
        examples=["Why Use Git When You Can Email Zip Files?"],
    )
    url: HttpUrl = Field(
        ...,
        examples=["https://supermuel.fr/why-use-git-when-you-can-email-zip-files"],
    )

    summary: str | None = Field(
        None,
        examples=[
            "This article is about why you should use Git instead of emailing zip files. The author explains the benefits of using Git and how it can improve your workflow."
        ],
    )

    # TODO : publish date
    # TODO : language

    class Settings:
        name = "articles"
        indexes = [
            IndexModel("url", unique=True),
        ]


class RSSFeedAnalysisStatus(str, Enum):
    """
    Enumeration representing the analysis status of an RSS feed.

    The status can be one of the following:
    - 'pending': The RSS feed has been added and is waiting to be analyzed.
    - 'in_progress': The analysis of the RSS feed is currently ongoing.
    - 'done': The analysis of the RSS feed has been completed successfully.
    - 'failed': The analysis of the RSS feed has failed.

    This status is used to manage the analysis process and prevent multiple concurrent analyses of the same RSS feed.
    """

    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"


class RSSFeed(Document):
    url: HttpUrl = Field(..., examples=["https://tim-tek.com/feed"])

    analysis_status: RSSFeedAnalysisStatus = Field(
        RSSFeedAnalysisStatus.pending,
        examples=["pending"],
    )

    class Settings:
        name = "rss_feeds"
        indexes = [
            IndexModel("url", unique=True),
        ]
