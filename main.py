from bson import ObjectId
from fastapi import FastAPI, HTTPException
from models import ArticleModel
from dotenv import load_dotenv
import os
from motor import motor_asyncio


load_dotenv()

client = motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.get_database("articles")
articles_collection = db.get_collection("articles")

app = FastAPI(
    title="BlogDB",
    summary="This is a simple API that serves articles summaries.",
    version="0.0.1",
    contact={
        "name": "SuperMuel",
        "url": "https://github.com/SuperMuel",
    },
)


@app.get(
    "/articles/",
    response_model=list[ArticleModel],
    response_model_by_alias=False,
)
async def read_articles():
    return await articles_collection.find().to_list(100)


@app.get(
    "/articles/{article_id}",
    response_model=ArticleModel,
    response_model_by_alias=False,
)
async def read_article(article_id: str):
    if (
        article := await articles_collection.find_one({"_id": ObjectId(article_id)})
    ) is not None:
        return article

    raise HTTPException(status_code=404, detail=f"Article {id} not found")
