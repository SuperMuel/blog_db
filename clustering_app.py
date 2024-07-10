import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import hdbscan
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import textwrap
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from textwrap import dedent
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
from pinecone.grpc import PineconeGRPC as Pinecone
from tqdm import tqdm

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["blogdb"]
collection = db["ai_news"]

# Pinecone setup
pc = Pinecone()
INDEX_NAME = "ai-news-index"
index = pc.Index(INDEX_NAME)


# Helper functions
def wrap_text(text, width=50):
    return "<br>".join(textwrap.wrap(text, width=width))


def get_cluster_center(points):
    return np.mean(points, axis=0)


def get_closest_points(points, center, n=5):
    distances = euclidean_distances([center], points)[0]  # type:ignore
    closest_indices = np.argsort(distances)[:n]
    return closest_indices


@st.cache_data(show_spinner=False)
def generate_cluster_title(article_titles: list[str]) -> str:
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307")  # type: ignore
    prompt = dedent("""Given the following news article titles, generate a single 3-4 words title that summarizes what the cluster of articles is about:

                        <titles>
                        {titles}
                        </titles>

                        Summary title:""")
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm | StrOutputParser()
    titles = "\n".join(f"- {title}" for title in article_titles[:5])
    return chain.invoke({"titles": titles})


# Streamlit app
st.set_page_config(
    page_title="AI News Cluster Visualization", page_icon="📰", layout="wide"
)
st.title("AI News Cluster Visualization")


# Sidebar controls
st.sidebar.header("Filters")
today = datetime.now().date()
default_start_date = today - timedelta(days=2)

dates = st.sidebar.date_input(
    "Date Range", [default_start_date, today], key="date_range"
)
assert isinstance(dates, tuple) and len(dates) == 2, "Invalid date range selected."

start_date, end_date = dates


st.sidebar.header("Clustering Parameters")
min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 20, 6)
min_samples = st.sidebar.slider("Min Samples", 2, 20, 3)

ORIGINAL_DIM = 1024
n_components = st.sidebar.slider("Number of dimensions", 2, ORIGINAL_DIM, ORIGINAL_DIM)


@st.cache_data(show_spinner="Loading data...")
def load_data(start_date, end_date):
    # Convert dates to datetime objects for MongoDB query
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Fetch IDs from MongoDB with date filter
    all_ids = [
        str(x["_id"])
        for x in collection.find(
            {"date": {"$gte": start_datetime, "$lte": end_datetime}}, {"_id": 1}
        )
    ]

    def fetch_all_vectors(index, all_ids: list[str], batch_size: int = 1000):
        all_data = []
        total_batches = (len(all_ids) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(all_ids), batch_size),
            total=total_batches,
            desc="Fetching vectors",
        ):
            batch_ids = all_ids[i : i + batch_size]
            batch_data = index.fetch(ids=batch_ids)
            all_data.extend(batch_data["vectors"].values())
        return all_data

    all_vectors = fetch_all_vectors(index, all_ids)
    data_dict = [
        {"id": x["id"], **x["metadata"], "embedding": x["values"]} for x in all_vectors
    ]

    for x in data_dict:
        x["found_at"] = datetime.fromtimestamp(x["found_at"])
        x["date"] = datetime.fromtimestamp(x["date"])

    df = pd.DataFrame(data_dict)
    return df


@st.cache_data(show_spinner="Clustering data...")
def perform_clustering(df, min_cluster_size, min_samples, n_components):
    print("Performing clustering...")
    matrix = np.array(df.embedding.tolist())

    # Perform dimensionality reduction
    if n_components < matrix.shape[1]:
        pca = PCA(n_components=n_components)
        reduced_matrix = pca.fit_transform(matrix)
    else:
        reduced_matrix = matrix

    # Perform clustering on reduced matrix
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    cluster_labels = clusterer.fit_predict(reduced_matrix)

    # Perform t-SNE on the reduced matrix for visualization
    tsne = TSNE(
        n_components=2, perplexity=15, random_state=42, init="random", learning_rate=150
    )
    vis_dims = tsne.fit_transform(reduced_matrix)

    tsne_df = pd.DataFrame(vis_dims, columns=["tsne_1", "tsne_2"])
    tsne_df["title"] = df["title"]
    tsne_df["wrapped_body"] = df["body"].apply(wrap_text)
    tsne_df["cluster"] = cluster_labels

    cluster_points = defaultdict(list)
    cluster_titles = defaultdict(list)
    for cluster, point, title in zip(
        tsne_df["cluster"], reduced_matrix, tsne_df["title"]
    ):
        cluster_points[cluster].append(point)
        cluster_titles[cluster].append(title)

    cluster_names = {-1: "Noise"}
    for cluster, points in cluster_points.items():
        if cluster != -1:
            center = get_cluster_center(points)
            closest_indices = get_closest_points(points, center)
            closest_titles = [cluster_titles[cluster][i] for i in closest_indices]
            cluster_names[cluster] = generate_cluster_title(closest_titles)

    tsne_df["cluster_name"] = tsne_df["cluster"].map(cluster_names)
    tsne_df["cluster"] = tsne_df["cluster_name"]
    tsne_df = tsne_df.drop("cluster_name", axis=1)

    print("Clustering completed!")
    return tsne_df, cluster_names


# Load data
df = load_data(start_date, end_date)

st.sidebar.write(f"Number of articles: {len(df)}")

# Perform clustering
tsne_df, cluster_names = perform_clustering(
    df, min_cluster_size, min_samples, n_components=n_components
)

fig = px.scatter(
    tsne_df,
    x="tsne_1",
    y="tsne_2",
    color="cluster",
    hover_data=["title", "wrapped_body", "cluster"],
    title=f"2D t-SNE projection of news articles with HDBSCAN clustering (Reduced to {n_components} dimensions)",
    labels={"tsne_1": "t-SNE feature 1", "tsne_2": "t-SNE feature 2"},
    color_discrete_sequence=px.colors.qualitative.Plotly,
    height=1000,
)

fig.update_traces(
    hovertemplate="<b>Title:</b> %{customdata[0]}<br><br>"
    "<b>Body:</b> %{customdata[1]}<br><br>"
    "<b>Cluster:</b> %{customdata[2]}"
)

fig.update_layout(hoverdistance=100, hovermode="closest")

st.plotly_chart(fig, use_container_width=True)

# Display cluster information
st.header("Cluster Information")
n_clusters = len(set(tsne_df["cluster"])) - 1  # Exclude 'Noise' cluster
st.write(f"Number of clusters: {n_clusters}")

# Display articles in the 'Noise' cluster
st.subheader("Articles in 'Noise' Cluster")
noise_articles = tsne_df[tsne_df["cluster"] == "Noise"]
st.write(
    f"Number of articles in 'Noise' cluster: {len(noise_articles)} ({(len(noise_articles) / len(tsne_df)) * 100:.2f}% of total)"
)
st.write("Sample of articles in 'Noise' cluster:")
st.write(noise_articles["title"].head(10).tolist())
