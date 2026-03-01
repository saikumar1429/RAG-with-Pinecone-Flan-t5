import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from transformers import pipeline


PINECONE_API_KEY = "pcsk_6srVnS_6GCXMTgsAPkTytddZnPmBQUob5MGHussED57QEH6LkbWhAqw6iWnCqogn9akNQ4"
INDEX_NAME = "rag-demo"
ENVIRONMENT = "https://rag-demo-umx5ffb.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1"
GENERATION_MODEL_NAME = "google/flan-t5-base"
TOP_K = 3
 
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
def load_generator():
    return pipeline("text2text-generation", model=GENERATION_MODEL_NAME)
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)
def generate_answer(query,index,embedder,generator):
    query_vector=embedder.encode(query).tolist()
    results= index.query(vector=query_vector,top_k=TOP_K,include_metadata=True)
    if not results["matches"]:
        return "No relevant context found.", ""

    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    prompt = f"""Context:
{context}

question:{query}
answer:"""
    output=generator(prompt)[0]['generated_text']
    return output,context
def main():
    st.set_page_config(page_title="RAG with Pinecone & Flan-T5", layout="wide")
    st.title("📚 Retrieval-Augmented Generation (RAG)")
    st.markdown("Ask a question and get answers using Flan-T5 with Pinecone vector search.")

    query = st.text_input("🔎 Enter your question here:")

    if query:
        with st.spinner("Searching and generating answer..."):
            embedder = load_embedder()
            generator = load_generator()
            index = init_pinecone()

            answer, context = generate_answer(query, index, embedder, generator)

        st.subheader("✅ Generated Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Context"):
            st.code(context)

# Run the app
if __name__ == "__main__":
    main()
