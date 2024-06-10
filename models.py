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

    class Settings:
        name = "articles"
        indexes = [
            IndexModel("url", unique=True),
        ]
