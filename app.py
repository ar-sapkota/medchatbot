import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Nepal Policy Q&A Bot", layout="centered")
st.title("📄 Nepal Policy Document Q&A Chatbot")
st.caption("Ask any question about Nepal's national interests, foreign policy, and strategic affairs")

# ==================== LOAD RESOURCES (CACHED) ====================
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])  # Use secrets (recommended)
    index = pc.Index("langchainvector")
    embeddings = load_embedding_model()
    return PineconeVectorStore(index=index, embedding=embeddings, namespace="default")

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",  # Your powerful free model
        temperature=0.3,
        api_key=st.secrets["GROQ_API_KEY"]
    )

@st.cache_resource
def load_rag_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context. Be clear, accurate, and professional.
If you don't know, say "I don't have information on that in the document."

Context:
{context}

Question: {input}
""")

    llm = load_llm()
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return rag_chain

# Load the chain
rag_chain = load_rag_chain()

# ==================== CHAT INTERFACE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    st.caption(f"Source {i}")
                    st.write(doc.page_content.strip()[:600] + "...")

# User input
if prompt := st.chat_input("Ask a question about Nepal's foreign policy, national interests, or strategic affairs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                sources = response["context"]

                st.markdown(answer)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error("Sorry, something went wrong. Please try again.")
                st.caption(f"Error: {e}")