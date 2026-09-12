import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.set_page_config(page_title="Book Companion AI", layout="wide")
    st.header("📚 Chat & Expand on Your PDF Book")

    # Sidebar inputs
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.sidebar.file_uploader("Upload your Book (PDF)", type=["pdf"])

    if uploaded_file and api_key:
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        # Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)

        # Vector store & Retriever
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        retriever = vector_store.as_retriever()

        # LLM Engine
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)

        # Prompt setup
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for answering questions about a book. Use the provided context to answer questions and expand on concepts. Context:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Modern RAG chain (LCEL)
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Session state for message history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # User input handling
        if user_input := st.chat_input("Ask a question or expand on an idea..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            # Format history for prompt
            formatted_history = []
            for m in st.session_state.messages[:-1]:
                role = "human" if m["role"] == "user" else "ai"
                formatted_history.append((role, m["content"]))

            # Generate output
            response = rag_chain.invoke({
                "question": user_input,
                "chat_history": formatted_history
            })

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
