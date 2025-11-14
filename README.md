# AmbedkarGPT

A simple Retrieval-Augmented Generation (RAG) prototype built using **LangChain**, **ChromaDB**, and **Ollama**.
This tool answers questions based on an excerpt from Dr. B.R. Ambedkar’s *Annihilation of Caste*.

This assignment demonstrates the core components of a RAG system:
loading text, chunking, generating embeddings, storing them in a vector database, retrieving relevant chunks, and generating answers using a local LLM.

---

## 🔧 Prerequisites

Before running this project, ensure you have the following installed:

* **Python 3.8+**
* **Ollama** → [https://ollama.ai](https://ollama.ai)
* An Ollama-compatible local LLM (e.g., `mistral` or a lightweight alternative)

  ```bash
  ollama pull mistral
  ```
* A Python virtual environment (recommended)

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Mauryalalit/AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Ensure `speech.txt` is present in the project directory (already included).
2. Make sure Ollama is running (`ollama serve` usually starts automatically).
3. Run the CLI application:

   ```bash
   python main.py
   ```
4. Example question:

   ```
   Why does Ambedkar criticize the shastras?
   ```
5. Type `exit` to quit the program.

---

## 🧠 How the System Works (RAG Pipeline Overview)

### 1. **Text Loading**

`TextLoader` reads the text from `speech.txt`.

### 2. **Chunking**

The text is split into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve context.

### 3. **Embeddings Generation**

Each chunk is converted into vector embeddings using:

* `sentence-transformers/all-MiniLM-L6-v2`

### 4. **Vector Store (ChromaDB)**

The embedding vectors are stored in a local ChromaDB instance for fast semantic search.

### 5. **Retriever**

The system finds the most relevant chunks based on the user’s query.

### 6. **LLM-Based Answer Generation**

Retrieved context + user question is passed to a local LLM (via Ollama), which generates the final answer.

---

## 📁 Repository Contents

* `main.py` – CLI application orchestrating the RAG workflow.
* `speech.txt` – Source text for question answering.
* `requirements.txt` – Python dependencies.
* `README.md` – Project documentation.

---

![Alt text](images/sample output.png)
