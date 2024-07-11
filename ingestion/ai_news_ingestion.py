import os
from pinecone.grpc import PineconeGRPC as Pinecone
import time
from typing import Sequence
from duckduckgo_search.exceptions import RatelimitException
import logging
import pymongo
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
from datetime import datetime
import pytz
from tqdm import tqdm
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from collections import defaultdict
import voyageai

from ingestion.configurations import Configuration, read_configs


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("ai_news_ingestion.log"),  # Log to file
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

CONFIGURATIONS_FILE = os.getenv("CONFIGURATIONS_FILE", "configs.json")
assert os.path.exists(
    CONFIGURATIONS_FILE
), f"Configurations file not found: {CONFIGURATIONS_FILE}"
logger.info(f"Using configurations file: {CONFIGURATIONS_FILE}")


def get_mongodb_collection():
    # MongoDB setup
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

    assert (
        MONGODB_URI and MONGODB_DATABASE and MONGODB_COLLECTION
    ), "MongoDB environment variables are not set."

    mongo_client = pymongo.MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DATABASE]
    return db[MONGODB_COLLECTION]


collection = get_mongodb_collection()

ARTICLE_TITLE_LENGTH_LIMIT = int(os.getenv("ARTICLE_TITLE_LENGTH_LIMIT", 200))
ARTICLE_BODY_LENGTH_LIMIT = int(os.getenv("ARTICLE_BODY_LENGTH_LIMIT", 1000))


# Voyageai setup
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
assert VOYAGE_API_KEY, "VOYAGE_API_KEY environment variable is not set."
assert EMBEDDING_MODEL, "EMBEDDING_MODEL environment variable is not set."
DIMENSIONS = os.getenv("DIMENSIONS")
assert DIMENSIONS, "DIMENSIONS environment variable is not set."
DIMENSIONS = int(DIMENSIONS)
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# Pinecone setup
pc = Pinecone()
assert os.getenv(
    "PINECONE_API_KEY"
), "PINECONE_API_KEY environment variable is not set."
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
assert INDEX_NAME, "PINECONE_INDEX_NAME environment variable is not set."
assert (
    INDEX_NAME in pc.list_indexes().names()
), f"Index {INDEX_NAME} not found in Pinecone."
pinecone_index = pc.Index(INDEX_NAME)


# Constants
INITIAL_SLEEP_TIME = int(os.getenv("INITIAL_SLEEP_TIME", 5))
MAX_SLEEP_TIME = int(os.getenv("MAX_SLEEP_TIME", 60))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 50))
TIME_LIMIT = os.getenv("TIME_LIMIT", "d")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
TIMEOUT = int(os.getenv("TIMEOUT", 30))


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=INITIAL_SLEEP_TIME, max=MAX_SLEEP_TIME),
)
def search_with_retry(
    ddgs: DDGS, query: str, max_results: int, timelimit: str, region: str = "wt-wt"
):
    try:
        return list(
            ddgs.news(
                query, max_results=max_results, timelimit=timelimit, region=region
            )
        )
    except RatelimitException:
        print(f"Rate limit hit for query: {query}. Retrying...")
        raise
    except requests.exceptions.Timeout:
        print(f"Timeout occurred for query: {query}. Retrying...")
        raise
    except Exception as e:
        print(f"Unexpected error occurred for query: {query}: {str(e)}. Retrying...")
        raise


def perform_search(
    queries: Sequence[str],
    region: str = "wt-wt",
    max_results: int = MAX_RESULTS,
    time_limit: str = TIME_LIMIT,
):
    results = {}
    duplicates = defaultdict(int)
    ddgs = DDGS(timeout=TIMEOUT)

    try:
        with tqdm(total=len(queries), desc="Searching") as pbar:
            for query in queries:
                pbar.set_postfix(query=query)
                try:
                    search_results = search_with_retry(
                        ddgs,
                        query,
                        max_results=max_results,
                        timelimit=time_limit,
                        region=region,
                    )
                    for result in search_results:
                        result["region"] = region

                        url = result["url"]
                        if url not in results:
                            result["found_at"] = datetime.now(pytz.utc).isoformat()
                            results[url] = result
                        else:
                            duplicates[url] += 1
                except Exception as e:
                    print(
                        f"Failed to complete search for query: {query}. Error: {str(e)}"
                    )
                finally:
                    time.sleep(INITIAL_SLEEP_TIME)
                    pbar.update(1)
    except KeyboardInterrupt:
        print("Search interrupted, returning results so far...")

    return results, duplicates


def get_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    result = vo.embed(texts, model=EMBEDDING_MODEL, input_type=input_type)
    return result.embeddings


def get_embeddings_batched(
    texts: list[str], batch_size: int = 128, input_type: str = "document"
) -> list[list[float]]:
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i : i + batch_size]
        batch_embeddings = get_embeddings(batch, input_type=input_type)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def upsert_to_pinecone(
    index,
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    batch_size: int = 100,
):
    assert all(len(vector) == DIMENSIONS for vector in embeddings)
    assert len(ids) == len(embeddings) == len(metadatas)

    vectors = [
        (id, embedding, metadata)
        for id, embedding, metadata in zip(ids, embeddings, metadatas)
    ]

    index.upsert(vectors=vectors, batch_size=batch_size)


def process_and_upsert_articles(index):
    query = {
        "$or": [{"pinecone_indexed": {"$exists": False}}, {"pinecone_indexed": False}]
    }
    articles = list(collection.find(query))

    if not articles:
        logger.info("No new articles to index.")
        return None

    logger.info(f"Indexing {len(articles)} new articles in Pinecone.")

    ids = [str(article["_id"]) for article in articles]
    texts = [f"{article['title']}\n\n{article['body']}" for article in articles]
    embeddings = get_embeddings_batched(texts=texts, input_type="document")

    metadatas = [
        {
            "title": article["title"],
            "url": article["url"],
            "body": article["body"],
            "found_at": article["found_at"].timestamp(),
            "date": article["date"].timestamp(),
        }
        for article in articles
    ]

    upsert_to_pinecone(index, ids, embeddings, metadatas)

    bulk_operations = [
        UpdateOne({"_id": article["_id"]}, {"$set": {"pinecone_indexed": True}})
        for article in articles
    ]
    collection.bulk_write(bulk_operations)

    logger.info(f"Indexed {len(articles)} new articles in Pinecone.")


def handle_configuration(config: Configuration):
    logger.info(f"Processing configuration: {config.name}")

    # Perform search for this configuration
    results, duplicates = perform_search(
        queries=config.queries,
        region=config.region,
        max_results=MAX_RESULTS,
        time_limit=TIME_LIMIT,
    )

    logger.info(f"Found {len(results)} unique results for {config.name}")
    logger.info(f"Found {len(duplicates)} duplicate URLs for {config.name}")
    logger.info(f"Total number of duplicate instances: {sum(duplicates.values())}")

    # Process and insert results into MongoDB
    processed_results = []
    for result in results.values():
        if isinstance(result["date"], str):
            result["date"] = datetime.fromisoformat(
                result["date"].replace("Z", "+00:00")
            )
        if isinstance(result["found_at"], str):
            result["found_at"] = datetime.fromisoformat(
                result["found_at"].replace("Z", "+00:00")
            )

        if len(result["title"]) > ARTICLE_TITLE_LENGTH_LIMIT:
            result["title"] = result["title"][:ARTICLE_TITLE_LENGTH_LIMIT]
        if len(result["body"]) > ARTICLE_BODY_LENGTH_LIMIT:
            result["body"] = result["body"][:ARTICLE_BODY_LENGTH_LIMIT]

        processed_results.append(result)

    if processed_results:
        try:
            insertion_result = collection.insert_many(processed_results, ordered=False)
            logger.info(
                f"Inserted {len(insertion_result.inserted_ids)} documents into MongoDB for {config.name}"
            )
        except BulkWriteError as e:
            logger.info(
                f"Inserted {e.details['nInserted']} documents into MongoDB for {config.name}"
            )
            logger.warning(
                f"Encountered {len(e.details['writeErrors'])} errors for {config.name}. (Duplicate URLs)"
            )
    else:
        logger.info(f"No new results to insert for {config.name}")

    # Process and upsert articles to Pinecone
    process_and_upsert_articles(pinecone_index)

    logger.info(f"Completed processing for configuration: {config.name}")


def main():
    logger.info("Starting AI news ingestion process...")

    # Read configurations
    all_configs = read_configs(CONFIGURATIONS_FILE)

    logger.info(f"Found {len(all_configs)} configurations")

    for config in all_configs:
        handle_configuration(config)

    logger.info("AI news ingestion process completed.")


if __name__ == "__main__":
    main()
