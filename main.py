import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from textwrap import indent

from beanie import PydanticObjectId, init_beanie
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from fastapi_utilities import repeat_every
from langchain_openai import ChatOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_core import Url

from models import Article, RSSFeed, RSSFeedAnalysisStatus, User
from rss_feed_parser import RSSFeedParser
from summarizer import ChainOfDensity

api_key_header = APIKeyHeader(name="X-API-Key")


logger = logging.getLogger(__name__)

# https://stackoverflow.com/a/77007723
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncIOMotorClient(os.environ["MONGODB_URI"])

    await init_beanie(
        database=client["blogdb"], document_models=[Article, User, RSSFeed]
    )
    logger.info("Connected to MongoDB")

    # Start the fetch_and_process_all_rss task in the background
    asyncio.create_task(fetch_and_process_all_rss())

    yield

    client.close()


app = FastAPI(
    title="BlogDB",
    summary="A microservice designed to monitor RSS feeds, generate concise summaries of new articles, store them in a database, and serve them via an API.",
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
    background_tasks.add_task(process_rss_feed, rss_feed)
    logger.info(f"RSS feed added: {rss_feed.url}, starting processing in background")
    return rss_feed


async def summarize(content: str) -> str:
    if not content:
        logger.warning("Empty content provided for summarization")
        raise ValueError("Empty content provided for summarization")

    model = ChatOpenAI(model="gpt-4o")
    cod = ChainOfDensity(model=model)

    try:
        summary = cod.generate_summary(content)
        return summary
    except Exception as e:
        logger.error(f"Failed to summarize content: {e}")
        raise e


# TODO : test with https://lorem-rss.herokuapp.com/
async def process_rss_feed(rss_feed: RSSFeed):
    if rss_feed.analysis_status == RSSFeedAnalysisStatus.in_progress:
        logger.info(f"RSS feed {rss_feed.url} is already being processed")
        return

    rss_feed.analysis_status = (
        RSSFeedAnalysisStatus.in_progress
    )  # Use an async context manager to ensure that the state is updated at the end
    await rss_feed.replace()

    logger.info(f"Processing RSS feed: {rss_feed.url}")

    try:
        parsed_feed = RSSFeedParser(rss_feed.url)

        logger.info(
            f"Found {parsed_feed.get_number_of_entries()} entries in feed {rss_feed.url}"
        )

        for entry in parsed_feed.get_entries():
            logger.info(f"Processing entry: {entry.title}")
            if not await Article.find_one({"url": entry.url}):
                logger.info(f"New article detected: {entry.title}")

                try:
                    logger.info(f"Summarizing article: {entry.title}")
                    summary = await summarize(entry.markdown_content)
                    logger.info(
                        f"Summary generated for article: {entry.title} \n{indent(summary, '    ')}"
                    )
                    article = Article(
                        title=entry.title, url=Url(entry.url), summary=summary
                    )
                    await article.insert()
                except Exception as e:  # TODO : better error handling
                    logger.error(f"Failed to summarize article: {entry.title}, {e}")
                    continue
                logger.info(f"New article added: {entry.title}")
            else:
                logger.info(f"Article already exists: {entry.title}")
    except Exception as e:
        logger.error(f"Failed to process RSS feed {rss_feed.url}: {e}")
        rss_feed.analysis_status = (
            RSSFeedAnalysisStatus.failed
        )  # TODO : add error message
        await rss_feed.replace()
        return

    logger.info(f"Finished processing RSS feed: {rss_feed.url}\n\n")
    rss_feed.analysis_status = RSSFeedAnalysisStatus.done
    await rss_feed.replace()


@repeat_every(seconds=int(os.environ.get("RSS_FEED_PROCESS_INTERVAL_SECONDS", 3600)))
async def fetch_and_process_all_rss():
    logger.info("Fetching and processing all RSS feeds")

    rss_feeds = await RSSFeed.find_all().to_list()

    if not rss_feeds:
        logger.info("No RSS feeds found")
        return

    for feed in rss_feeds:
        await process_rss_feed(feed)
