import streamlit as st
from datetime import datetime
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import format_document
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from datetime import date
from dotenv import load_dotenv
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


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
    "<question>\n"
    "{input}\n"
    "</question>\n\n"
    "<context>\n"
    "{context}\n"
    "</context>\n\n"
    "Answer:"
)


# Helper functions
def format_docs(docs: list[Document], separator: str = "\n\n") -> str:
    doc_prompt = PromptTemplate.from_template(
        "{page_content} - (Published on {date})\n{url}\n{body}"
    )
    return separator.join(format_document(doc, doc_prompt) for doc in docs)


def convert_docs_dates(docs: list[Document]) -> list[Document]:
    """Convert Unix timestamps to human-readable dates in document metadata."""
    for doc in docs:
        for key in ["date", "found_at"]:
            if key in doc.metadata:
                doc.metadata[key] = datetime.fromtimestamp(doc.metadata[key]).strftime(
                    "%Y-%m-%d"
                )
    return docs


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    unique_docs = {}
    for doc in docs:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
    return list(unique_docs.values())


history = StreamlitChatMessageHistory(key="history")

# Set up retriever and RAG chain
retriever = docsearch.as_retriever(search_kwargs={"k": 30})

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


rag_chain = (
    {
        "input": RunnablePassthrough(),
        "context": history_aware_retriever
        | convert_docs_dates
        | deduplicate_docs
        | format_docs,
    }
    | prompt
    | llm
    | StrOutputParser()
)

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)


# Streamlit UI
st.write("Ask me anything about the latest AI news!")


# Display chat messages from history on app rerun
for msg in history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# React to user input
if prompt := st.chat_input("What would you like to know about AI?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for chunk in chain_with_history.stream(
            {"input": prompt},
            config={"configurable": {"session_id": "any"}},
        ):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    print(
        f"At the end of the assistant answer, last message from history is : {history.messages[-1]}"
    )

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
