# Mini RAG Project

This project is a minimal Retrieval-Augmented Generation (RAG) system built with Python, Sentence Transformers, cosine similarity retrieval, and a local Large Language Model through Ollama.

The goal of this project is to understand the complete workflow of a basic RAG system, including document chunking, embedding generation, semantic retrieval, prompt construction, LLM-based answer generation, source citation, and query history logging.

---

## Project Goal

The project answers user questions based on a small local knowledge base.

Instead of relying only on the language model's internal knowledge, the system first retrieves relevant text chunks from `knowledge.txt`, then sends the retrieved context to a local LLM to generate a grounded answer.

---

## What Is RAG?

RAG stands for Retrieval-Augmented Generation.

A RAG system usually contains two main parts:

1. **Retriever**: Finds relevant information from a knowledge base.
2. **Generator**: Uses the retrieved information to generate an answer.

In this project:

- `build_index.py` works as the index builder.
- `retrieve_top_k()` works as the retriever.
- `generate_answer()` works as the generator.
- Ollama with `qwen2.5:7b` works as the local LLM.

---

## Project Structure

```text
mini_rag_project/
│
├── data/
│   └── knowledge.txt
│
├── outputs/
│   ├── chunks.pkl
│   ├── chunk_embeddings.pkl
│   ├── last_query_result.txt
│   └── query_history.jsonl
│
├── src/
│   ├── build_index.py
│   ├── rag_query.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Core Workflow

```text
Local Knowledge File
        ↓
Split Text into Chunks
        ↓
Generate Chunk Embeddings
        ↓
Save Chunks and Embeddings
        ↓
User Query
        ↓
Generate Query Embedding
        ↓
Cosine Similarity Search
        ↓
Retrieve Top Relevant Chunks
        ↓
Filter Weakly Related Chunks
        ↓
Build Prompt with Retrieved Context
        ↓
Local LLM via Ollama
        ↓
Generate Final Answer
        ↓
Show Source Citations and Save Query History
```

---

## Features

This project includes:

1. Document chunking
2. Sentence embedding generation
3. Query embedding generation
4. Cosine similarity retrieval
5. Top-k relevant chunk selection
6. Dynamic similarity threshold filtering
7. Local LLM answer generation using Ollama
8. Source citation display
9. Multi-turn command-line interaction
10. Query history logging in JSONL format

---

## Technologies Used

- Python
- Sentence Transformers
- scikit-learn
- Ollama
- Qwen2.5 7B
- Cosine Similarity
- JSONL logging

---

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and pull the local model:

```bash
ollama pull qwen2.5:7b
```

Check downloaded models:

```bash
ollama list
```

---

## How to Run

### Step 1: Build the index

```bash
python src/build_index.py
```

This step reads the knowledge file, splits it into chunks, generates embeddings, and saves the results to the `outputs/` folder.

### Step 2: Start the RAG query system

```bash
python src/rag_query.py
```

Then enter questions in the terminal.

Example:

```text
请输入你的问题: what is RAG?
```

To exit the system:

```text
exit
```

---

## Example Output

Example question:

```text
what is transformer?
```

Example answer:

```text
Transformer 是基于自注意力机制的模型。注意力机制帮助模型聚焦于输入中最相关部分。
通过 Query、Key 和 Value 来计算注意力。多头注意力机制使模型可以从多个角度学习。

参考来源：
[1] score=0.6284
Transformer is based on self-attention.
Attention helps the model focus on the most relevant parts of the input.
Query, Key, and Value are used to compute attention.
Multi-head attention allows the model to learn from multiple perspectives.
```

---

## Output Files

The system generates several output files:

- `outputs/chunks.pkl`: Stores the processed text chunks.
- `outputs/chunk_embeddings.pkl`: Stores the vector embeddings of all chunks.
- `outputs/last_query_result.txt`: Stores the latest RAG answer.
- `outputs/query_history.jsonl`: Stores all query history records.

---

## Query History Format

Each query record is saved as one JSON line:

```json
{
  "time": "2026-05-04T18:30:12",
  "query": "what is RAG?",
  "answer": "RAG 是 retrieval-augmented generation 的缩写...",
  "sources": [
    {
      "source_id": 1,
      "score": 0.3500,
      "chunk": "RAG means retrieval-augmented generation..."
    }
  ]
}
```

---

## What I Learned

Through this project, I learned:

1. How to split a knowledge file into text chunks.
2. How to convert text chunks into semantic embeddings.
3. How to convert a user query into an embedding.
4. How to use cosine similarity for semantic search.
5. How to retrieve the most relevant context.
6. How to filter weakly related chunks.
7. How to construct a RAG prompt.
8. How to use a local LLM through Ollama.
9. How to generate answers grounded in retrieved context.
10. How to add source citations and query history logging.

---

## Conclusion

This project demonstrates the complete workflow of a minimal local RAG system.

It shows how retrieval and generation can be combined to answer questions based on a local knowledge base. Compared with a normal LLM chatbot, this system can provide answers grounded in specific retrieved documents and display the source context used for generation.

## Current Version

This project is now a complete minimal local RAG system with semantic retrieval, dynamic chunk filtering, local LLM generation, source citation, multi-turn interaction, and query history logging.

## Highlights

- Built a complete minimal RAG pipeline from document chunking to LLM-based answer generation.
- Used Sentence Transformers to convert text chunks and user queries into semantic embeddings.
- Applied cosine similarity to retrieve the most relevant chunks from the local knowledge base.
- Added dynamic similarity filtering to reduce weakly related or irrelevant context.
- Integrated Ollama with Qwen2.5 for local LLM answer generation.
- Added source citation to show which chunks were used to generate the answer.
- Added JSONL-based query history logging for debugging and future analysis.