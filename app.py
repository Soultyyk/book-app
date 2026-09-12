import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def main():
    st.set_page_config(page_title="Book Companion AI", layout="wide")
    st.header("📚 Chat & Expand on Your PDF Book")

    # 1. API Key & File Upload
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.sidebar.file_uploader("Upload your Book (PDF)", type=["pdf"])

    if uploaded_file and api_key:
        # Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Chunk text for retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)

        # Create local vector database
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Initialize conversational chain with memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        # 2. Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask a question or brainstorm an expansion..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            response = qa_chain.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()