# RAG-
# 🧠 RAG Pipeline: A Quiet Conversation with Knowledge

This project is a small act of care.

It reads a 3-million-character medical encyclopedia, breaks it into thoughtful pieces, and stores them in a local memory—ready to answer questions like “What are the symptoms of schizophrenia?” without ever calling the cloud.

No billing. No rate limits. Just local embeddings, ChromaDB, and a desire to make knowledge feel close, searchable, and kind.

---

## 🌱 What it does

- Loads and chunks a full medical PDF using `PyMuPDF`
- Embeds each chunk using `all-MiniLM-L6-v2` (via `sentence-transformers`)
- Stores those embeddings in a local ChromaDB collection
- Lets you ask questions and retrieves the most relevant answers

---

## 💡 Why it matters

Because sometimes, the most powerful thing you can build is a tool that listens.

This pipeline is for anyone who wants to:
- Learn from large documents without sending data to the cloud
- Build beginner-friendly, offline-first RAG systems
- Leave behind a legacy of clarity, care, and contribution

---

## 🛠️ How to run

```bash
pip install -r requirements.txt
python app3.py
