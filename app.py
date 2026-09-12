import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        retriever = vector_store.as_retriever()

        # LLM Engine
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)

        # System Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks based on the provided book context.\n"
            "Use the following pieces of retrieved context to answer the user's questions and expand on ideas.\n"
            "If you don't know the answer, say that you don't know.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Chat memory setup using Streamlit session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display past messages
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Chat input box
        if user_input := st.chat_input("Ask a question or brainstorm an expansion..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            # Format history for prompt
            formatted_history = []
            for m in st.session_state.messages[:-1]:
                role = "human" if m["role"] == "user" else "ai"
                formatted_history.append((role, m["content"]))

            response = rag_chain.invoke({"input": user_input, "chat_history": formatted_history})
            answer = response["answer"]

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()
