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
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
import pymongo
from pinecone.grpc import PineconeGRPC as Pinecone
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

# Load environment variables
load_dotenv()


class DataLoader:
    def __init__(self):
        self.mongo_client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.mongo_client["blogdb"]
        self.collection = self.db["ai_news"]
        self.pc = Pinecone()
        self.index = self.pc.Index("ai-news-index")

    @st.cache_data(show_spinner="Loading data...")
    def load_data(_self, start_date: date, end_date: date) -> pd.DataFrame:
        assert (
            0 <= (end_date - start_date).days <= 365
        ), "Date range must be between 0 and 365 days."

        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        all_ids = [
            str(x["_id"])
            for x in _self.collection.find(
                {"date": {"$gte": start_datetime, "$lte": end_datetime}}, {"_id": 1}
            )
        ]

        all_vectors = _self._fetch_all_vectors(all_ids)
        data_dict = [
            {"id": x["id"], **x["metadata"], "embedding": x["values"]}
            for x in all_vectors
        ]

        for x in data_dict:
            x["found_at"] = datetime.fromtimestamp(x["found_at"])
            x["date"] = datetime.fromtimestamp(x["date"])

        return pd.DataFrame(data_dict)

    def _fetch_all_vectors(
        self, all_ids: List[str], batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        all_data = []
        total_batches = (len(all_ids) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(all_ids), batch_size),
            total=total_batches,
            desc="Fetching vectors",
        ):
            batch_ids = all_ids[i : i + batch_size]
            batch_data = self.index.fetch(ids=batch_ids)
            all_data.extend(batch_data["vectors"].values())
        return all_data


class ClusteringEngine:
    @staticmethod
    def get_cluster_center(points: np.ndarray) -> np.ndarray:
        return np.mean(points, axis=0)

    @staticmethod
    def get_closest_points(
        points: np.ndarray, center: np.ndarray, n: int = 5
    ) -> List[int]:
        distances = euclidean_distances([center], points)[0]  # type: ignore
        return np.argsort(distances)[:n].tolist()

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_cluster_title(article_titles: List[str]) -> str:
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

    @st.cache_data(show_spinner="Clustering data...")
    def perform_clustering(
        _self,
        df: pd.DataFrame,
        min_cluster_size: int,
        min_samples: int,
        n_components: int,
    ) -> Tuple[pd.DataFrame, Dict[int, str]]:
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
            n_components=2,
            perplexity=15,
            random_state=42,
            init="random",
            learning_rate=150,
        )
        vis_dims = tsne.fit_transform(reduced_matrix)

        tsne_df = pd.DataFrame(vis_dims, columns=["tsne_1", "tsne_2"])
        tsne_df["title"] = df["title"]
        tsne_df["body"] = df["body"]
        tsne_df["url"] = df["url"]
        tsne_df["found_at"] = df["found_at"]
        tsne_df["date"] = df["date"]

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
                center = _self.get_cluster_center(points)  # type: ignore
                closest_indices = _self.get_closest_points(points, center)  # type: ignore
                closest_titles = [cluster_titles[cluster][i] for i in closest_indices]
                cluster_names[cluster] = _self.generate_cluster_title(closest_titles)

        tsne_df["cluster_name"] = tsne_df["cluster"].map(cluster_names)
        tsne_df["cluster"] = tsne_df["cluster_name"]
        tsne_df = tsne_df.drop("cluster_name", axis=1)

        print("Clustering completed!")
        return tsne_df, cluster_names


class Visualizer:
    @staticmethod
    def wrap_text(text: str, width: int = 50) -> str:
        return "<br>".join(textwrap.wrap(text, width=width))

    @staticmethod
    def create_scatter_plot(tsne_df: pd.DataFrame, n_components: int):
        tsne_df["wrapped_body"] = tsne_df["body"].apply(Visualizer.wrap_text)

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
        return fig


class StreamlitApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.clustering_engine = ClusteringEngine()
        self.visualizer = Visualizer()

    def run(self):
        st.set_page_config(
            page_title="AI News Cluster Visualization", page_icon="📰", layout="wide"
        )
        st.title("AI News Cluster Visualization")

        self._setup_sidebar()
        df = self.data_loader.load_data(self.start_date, self.end_date)
        st.sidebar.write(f"Number of articles: {len(df)}")

        tsne_df, cluster_names = self.clustering_engine.perform_clustering(
            df, self.min_cluster_size, self.min_samples, n_components=self.n_components
        )

        fig = self.visualizer.create_scatter_plot(tsne_df, self.n_components)
        st.plotly_chart(fig, use_container_width=True)

        self._display_cluster_info(tsne_df, cluster_names)

    def _setup_sidebar(self):
        st.sidebar.header("Filters")
        today = datetime.now().date()
        default_start_date = today - timedelta(days=2)

        dates = st.sidebar.date_input(
            "Date Range", [default_start_date, today], key="date_range"
        )
        assert (
            isinstance(dates, tuple) and len(dates) == 2
        ), "Invalid date range selected."
        self.start_date, self.end_date = dates

        st.sidebar.header("Clustering Parameters")
        self.min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 20, 6)
        self.min_samples = st.sidebar.slider("Min Samples", 2, 20, 3)

        ORIGINAL_DIM = 1024
        self.n_components = st.sidebar.slider(
            "Number of dimensions",
            2,
            ORIGINAL_DIM,
            ORIGINAL_DIM,
            help="Reduce the dimensionality of the data before clustering. This can help improve performance.",
        )

    def _display_cluster_info(
        self, tsne_df: pd.DataFrame, cluster_names: Dict[int, str]
    ):
        st.header("Cluster Information")
        n_clusters = len(set(tsne_df["cluster"])) - 1  # Exclude 'Noise' cluster
        st.write(f"Number of clusters: {n_clusters}")

        st.subheader("Articles in 'Noise' Cluster")
        noise_articles = tsne_df[tsne_df["cluster"] == "Noise"]
        st.write(
            f"Number of articles in 'Noise' cluster: {len(noise_articles)} ({(len(noise_articles) / len(tsne_df)) * 100:.2f}% of total)"
        )
        st.write("Sample of articles in 'Noise' cluster:")
        st.write(noise_articles["title"].head(10).tolist())


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
