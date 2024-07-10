import streamlit as st
from datetime import date, datetime
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
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


class AINewsConfig:
    EMBEDDING_MODEL = "voyage-large-2-instruct"
    INDEX_NAME = "ai-news-index"
    LLM_MODEL = "claude-3-5-sonnet-20240620"


class DocumentProcessor:
    @staticmethod
    def format_docs(docs: list[Document], separator: str = "\n\n") -> str:
        doc_prompt = PromptTemplate.from_template(
            "{page_content} - (Published on {date})\n{url}\n{body}"
        )
        return separator.join(format_document(doc, doc_prompt) for doc in docs)

    @staticmethod
    def convert_docs_dates(docs: list[Document]) -> list[Document]:
        for doc in docs:
            for key in ["date", "found_at"]:
                if key in doc.metadata:
                    doc.metadata[key] = datetime.fromtimestamp(
                        doc.metadata[key]
                    ).strftime("%Y-%m-%d")
        return docs

    @staticmethod
    def deduplicate_docs(docs: list[Document]) -> list[Document]:
        return list({doc.page_content: doc for doc in docs}.values())


class AINewsPrompts:
    @staticmethod
    def get_qa_prompt():
        qa_system_prompt = (
            "You are an assistant for question-answering about the latest AI news. "
            "Use the following pieces of retrieved news articles to answer the question. "
            "You are talking to an experienced audience in AI. "
            "If you don't know the answer, just say that you don't know. "
            "Format your answer in markdown and add inline hyperlinks in your words or sentences."
            "Do not start your answer with 'Based on the provided context', or similar phrases. "
            f"Today is {date.today()} and below are the latest news on AI. \n"
            "<context>\n{context}\n</context>\n\n"
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    @staticmethod
    def get_contextualize_prompt():  # TODO : on long conversations, it fails to rewrite the question, and answers it instead.
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


class AINewsChatbot:
    def __init__(self):
        self.embeddings = VoyageAIEmbeddings(  # type: ignore
            model=AINewsConfig.EMBEDDING_MODEL, batch_size=128
        )
        self.docsearch = PineconeVectorStore(
            index_name=AINewsConfig.INDEX_NAME,
            embedding=self.embeddings,
            text_key="title",
        )
        self.llm = ChatAnthropic(model_name=AINewsConfig.LLM_MODEL, temperature=0.0)  # type: ignore
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.qa_prompt = AINewsPrompts.get_qa_prompt()
        self.contextualize_prompt = AINewsPrompts.get_contextualize_prompt()
        self.chain_with_history = self.setup_rag_chain()

    def setup_rag_chain(self):
        retriever = self.docsearch.as_retriever(search_kwargs={"k": 30})
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_prompt
        )

        rag_chain = (
            RunnablePassthrough.assign(
                context=history_aware_retriever
                | DocumentProcessor.convert_docs_dates
                | DocumentProcessor.deduplicate_docs
                | DocumentProcessor.format_docs
            )
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            rag_chain,  # type: ignore
            lambda session_id: self.history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def process_user_input(self, user_input: str):
        return self.chain_with_history.invoke(
            {"input": user_input}, config={"configurable": {"session_id": "any"}}
        )


class StreamlitUI:
    def __init__(self, chatbot: AINewsChatbot):
        self.chatbot = chatbot

    def setup_page(self):
        st.set_page_config(page_title="AI News Chatbot", page_icon="🤖", layout="wide")
        st.title("AI News Chatbot")
        st.write("Ask me anything about the latest AI news!")

    def setup_sidebar(self):
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

    def display_chat_history(self):
        for msg in self.chatbot.history.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

    def handle_user_input(self):
        if prompt := st.chat_input("What would you like to know about AI?"):
            st.chat_message("user").markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in self.chatbot.chain_with_history.stream(
                    {"input": prompt},
                    config={"configurable": {"session_id": "any"}},
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

    def run(self):
        self.setup_page()
        self.setup_sidebar()
        self.display_chat_history()
        self.handle_user_input()


def main():
    chatbot = AINewsChatbot()
    ui = StreamlitUI(chatbot)
    ui.run()


if __name__ == "__main__":
    main()
