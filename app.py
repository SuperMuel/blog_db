from datetime import date, datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    format_document,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings

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
qa_system_prompt = (
    "You are an assistant for question-answering about the latest AI news. "
    "Use the following pieces of retrieved news articles to answer the question. "
    "You are talking to an experienced audience in AI. "
    "If you don't know the answer, just say that you don't know. "
    "Format your answer in markdown and add inline hyperlinks in your words or sentences."
    "Do not start your answer with 'Based on the provided context', or similar phrases. "
    f"Today is {date.today()} and below are the latest news on AI. \n"
    "<context>\n"
    "{context}\n"
    "</context>\n\n"
)

qa_prompt = qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
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


history = StreamlitChatMessageHistory(key="chat_history")

# Set up retriever and RAG chain
retriever = docsearch.as_retriever(search_kwargs={"k": 30})

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


rag_chain = (
    RunnablePassthrough.assign(
        context=history_aware_retriever
        | convert_docs_dates
        | deduplicate_docs
        | format_docs
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

chain_with_history = RunnableWithMessageHistory(
    rag_chain,  # type: ignore
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
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
        # st.markdown(
        #     chain_with_history.invoke(
        #         {"input": prompt}, config={"configurable": {"session_id": "any"}}
        #     )
        # )


with st.sidebar:
    st.title("About")
    st.info(
        "This AI News Chatbot uses a RAG (Retrieval-Augmented Generation) system to answer "
        "questions about the latest AI news. It retrieves relevant information from a "
        "Pinecone vector database and generates responses using the Claude 3.5 Sonnet model."
    )
    st.title("Tips")
    st.info(
        "- Ask about recent AI developments, products, or news.\n"
        "- Be specific in your questions for more accurate answers.\n"
        "- The bot can provide information on AI companies, research, and technologies."
    )
