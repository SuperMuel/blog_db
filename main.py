import os
from contextlib import asynccontextmanager

from beanie import PydanticObjectId, init_beanie
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from motor.motor_asyncio import AsyncIOMotorClient

from models import Article, User

api_key_header = APIKeyHeader(name="X-API-Key")


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncIOMotorClient(os.environ["MONGODB_URI"])

    await init_beanie(database=client["blogdb"], document_models=[Article, User])

    yield

    client.close()


app = FastAPI(
    title="BlogDB",
    summary="This is a simple API that generates, stores and serves Blog articles summaries.",
    version="0.0.1",
    contact={
        "name": "SuperMuel",
        "url": "https://github.com/SuperMuel",
    },
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Welcome to BlogDB"}


# TODO : TLS
async def get_authenticated_user(api_key: str = Security(api_key_header)) -> User:
    user = await User.find_one({"api_key": api_key})

    if user is not None:
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )


@app.get(
    "/articles/",
    response_model=list[Article],
    response_model_by_alias=False,
)
async def read_articles(
    request: Request,
    user: str = Security(get_authenticated_user),
):
    return await Article.find_all().to_list(100)


@app.get(
    "/articles/{article_id}",
    response_model=Article,
    response_model_by_alias=False,
)
async def read_article(article_id: PydanticObjectId):
    article = await Article.get(article_id)

    if article is not None:
        return article

    raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
