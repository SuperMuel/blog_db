from pydantic import (
    BaseModel,
    Field,
    EmailStr,
    BeforeValidator,
    ConfigDict,
    UrlConstraints,
)

from typing import Annotated

from pydantic_core import Url


# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


HttpUrl = Annotated[
    Url,
    UrlConstraints(max_length=2083, allowed_schemes=["http", "https"]),
]


class ArticleModel(BaseModel):
    id: PyObjectId | None = Field(alias="_id", default=None)
    title: str = Field(
        ...,
        max_length=300,
        examples=["Why Use Git When You Can Email Zip Files?"],
    )
    url: HttpUrl = Field(
        ...,
        examples=["https://supermuel.fr/why-use-git-when-you-can-email-zip-files"],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
