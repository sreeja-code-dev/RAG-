import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# 🚀 Load environment variables
load_dotenv()

# 📄 Load and chunk PDF
pdf_path = r"C:\Users\SREEJA\Desktop\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
doc = fitz.open(pdf_path)
text_chunks = []

for i, page in enumerate(doc):
    text = page.get_text()
    if text:
        text_chunks.append(text)

print("PDF loaded. Total characters:", sum(len(chunk) for chunk in text_chunks))

# 🧠 Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast and reliable

def embed_text(text):
    return model.encode(text, convert_to_numpy=True).tolist()

# 🗃️ Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="gale-encyclopedia")

# 📦 Upsert chunks into ChromaDB
for i, chunk in enumerate(text_chunks):
    embedding = embed_text(chunk)
    collection.add(
        ids=[f"gale-doc-{i}"],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": "GALE Encyclopedia", "page": i}]
    )

# 🔍 Query ChromaDB for schizophrenia
query2 = "What are the symptoms of schizophrenia?"
query_embedding2 = embed_text(query2)

results2 = collection.query(
    query_embeddings=[query_embedding2],
    n_results=3,
    include=["documents", "metadatas"]
)

# 📤 Output results
for i, doc in enumerate(results2["documents"][0]):
    metadata = results2["metadatas"][0][i]
    print(f"\n🧠 Schizophrenia Result {i+1} (Page {metadata['page']}):\n{doc[:500]}...\n")