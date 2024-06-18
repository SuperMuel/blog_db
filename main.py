import os
from contextlib import asynccontextmanager

import feedparser
from beanie import PydanticObjectId, init_beanie
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from motor.motor_asyncio import AsyncIOMotorClient

from models import Article, RSSFeed, User

api_key_header = APIKeyHeader(name="X-API-Key")


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncIOMotorClient(os.environ["MONGODB_URI"])

    # TODO : start background rss feed processing

    await init_beanie(
        database=client["blogdb"], document_models=[Article, User, RSSFeed]
    )

    yield

    client.close()


app = FastAPI(
    title="BlogDB",
    summary="This is a simple API that generates, stores and serves Blog articles summaries.",
    version="0.0.2",
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
    tags=["articles"],
    operation_id="get_articles",
)
async def get_articles(
    request: Request,
    user: str = Security(get_authenticated_user),
):
    return await Article.find_all().to_list(100)


@app.get(
    "/articles/{article_id}",
    response_model=Article,
    tags=["articles"],
    operation_id="get_article",
)
async def read_article(article_id: PydanticObjectId):
    article = await Article.get(article_id)

    if article is not None:
        return article

    raise HTTPException(status_code=404, detail=f"Article {article_id} not found")


@app.post(
    "/rss_feeds/",
    response_model=RSSFeed,
    tags=["rss_feeds"],
    operation_id="add_rss_feed",
)
async def add_rss_feed(
    rss_feed: RSSFeed,
    background_tasks: BackgroundTasks,
    user: User = Security(get_authenticated_user),
):
    existing_feed = await RSSFeed.find_one({"url": rss_feed.url})
    if existing_feed:
        return existing_feed

    await rss_feed.insert()
    # background_tasks.add_task(process_rss_feed, rss_feed)
    return rss_feed


def summarize(url: str) -> str:
    # TODO
    return "Résumé de l'article généré par IA"


async def process_rss_feed(rss_feed: RSSFeed):
    rss_url = rss_feed.url
    parsed_feed = feedparser.parse(rss_url)
    for entry in parsed_feed.entries:
        if not await Article.find_one({"url": entry.link}):
            # New article detected
            summary = summarize(entry.url)
            article = Article(title=entry.title, url=entry.link, summary=summary)
            await article.insert()
            print(f"Nouvel article ajouté: {entry.title}")


async def fetch_and_process_all_rss():
    rss_feeds = await RSSFeed.find_all().to_list()
    for feed in rss_feeds:
        await process_rss_feed(feed)
