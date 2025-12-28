import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

st.set_page_config(page_title="PDF Q&A", layout="wide")

st.title("📄 PDF Question Answering")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    pc = Pinecone(api_key="pcsk_2LxUZh_RY3HRa6SpnnBSJEr8vDLnNCDQupA1XtrxYK1JCuP7Mg1c3dkPEnxAKPHKBh9EZM")
    return pc.Index("langchainvector")

model = load_model()
index = load_index()

query = st.text_input("Ask a question about the document:")

if query:
    with st.spinner("Searching..."):
        query_embedding = model.encode(query).tolist()

        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace="default"
        )

        st.subheader("🔍 Retrieved Context")
        for match in results["matches"]:
            st.write(match["metadata"]["text"][:500])
            st.write("Score:", match["score"])
            st.divider()
