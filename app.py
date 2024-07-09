import streamlit as st
from langchain_core.prompts import format_document
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from operator import itemgetter
from datetime import date
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="AI News Chatbot", page_icon="🤖", layout="wide")
st.title("AI News Chatbot")

# Initialize Pinecone and embeddings
EMBEDDING_MODEL = "voyage-large-2-instruct"
INDEX_NAME = "ai-news-index"

embeddings = VoyageAIEmbeddings(model=EMBEDDING_MODEL)  # type: ignore

docsearch = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="title",
)

# Initialize language model
llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0.0)  # type: ignore

# Define prompt template
prompt = PromptTemplate.from_template(
    "You are an assistant for question-answering about the latest AI news. "
    "Use the following pieces of retrieved news articles to answer the question. "
    "You are talking to an experienced audience in AI. "
    "If you don't know the answer, just say that you don't know. "
    "Format your answer in markdown and add inline hyperlinks."
    "Do not start your answer with 'Based on the provided context', or similar phrases. "
    f"Today is {date.today()} and below are the latest news on AI. \n"
    "Question : {question}\n"
    "Context : \n{context}\n\n"
    "Answer:"
)


# Helper functions
def format_docs(docs: list[Document], separator: str = "\n\n") -> str:
    doc_prompt = PromptTemplate.from_template(
        "{page_content} - (Published on {date})\n{url}\n{body}"
    )
    return separator.join(format_document(doc, doc_prompt) for doc in docs)


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    unique_docs = {}
    for doc in docs:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
    return list(unique_docs.values())


# Set up retriever and RAG chain
retriever = docsearch.as_retriever(search_kwargs={"k": 30})

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": itemgetter("question")
        | retriever
        | RunnableLambda(deduplicate_docs)
        | RunnableLambda(format_docs),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.write("Ask me anything about the latest AI news!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about AI?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for chunk in rag_chain.stream({"question": prompt}):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a sidebar with information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This AI News Chatbot uses a RAG (Retrieval-Augmented Generation) system to answer "
    "questions about the latest AI news. It retrieves relevant information from a "
    "Pinecone vector database and generates responses using the Claude 3.5 Sonnet model."
)
st.sidebar.title("Tips")
st.sidebar.info(
    "- Ask about recent AI developments, products, or news.\n"
    "- Be specific in your questions for more accurate answers.\n"
    "- The bot can provide information on AI companies, research, and technologies."
)
