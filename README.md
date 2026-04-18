# ⚡ InfoWave AI — Unified RAG for Documents & Web

InfoWave AI is a high-performance **Retrieval-Augmented Generation (RAG)** application that enables users to query **PDF documents and websites** with intelligent retrieval strategies and accurate, context-aware answers.

Built with **Streamlit, LangChain, Groq, Ollama, and FAISS**, this project focuses on **speed, scalability, and precision**.

---

## 🚀 Features

### 🔍 Multi-Source Querying
- 📄 Upload and analyze **PDF documents**
- 🌐 Extract and query **website content**

### 🧠 Smart Query Modes
- **📖 Full Read Mode (MapReduce)**
  - Smart sampling for large documents
  - Parallel chunk processing
  - Hierarchical summarization

- **🎯 Targeted Mode (MMR Retrieval)**
  - Fast similarity-based search
  - Reduces redundant context
  - Optimized for quick responses

---

### ⚡ Performance Optimizations
- ✅ FAISS **disk caching** (avoid re-embedding)
- ✅ Batched embedding (Ollama optimized)
- ✅ Smart chunking strategy
- ✅ Parallel MapReduce with rate-limit control

---

### 🔊 Text-to-Speech
- Convert generated answers into audio using **gTTS**

---

### 🎨 UI/UX
- Modern, responsive UI with custom CSS
- Clean document visualization
- Interactive retrieval insights

---

## 🏗️ Tech Stack

| Layer        | Technology |
|-------------|-----------|
| Frontend     | Streamlit |
| LLM          | Groq (LLaMA / Qwen) |
| Embeddings   | Ollama (`nomic-embed-text`) |
| Vector DB    | FAISS |
| PDF Parsing  | PyPDF |
| Web Loader   | LangChain |

---

## 📁 Project Structure
'''.
├── app.py                 # Main Streamlit application
├── .env                   # API keys
├── .faiss_cache/          # Cached vector indexes
├── README.md
'''

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/infowave-ai.git
cd infowave-ai
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Setup Environment Variables
Create a .env file:
```
GROQ_API_KEY=your_groq_api_key
```
## 🧠 Setup Ollama (Required)

Install and run Ollama:
```Bash
ollama pull nomic-embed-text
ollama serve
```
## ▶️ Run the Application
```
streamlit run app.py
```

## 🧪 How It Works
### 📄 PDF Workflow
1. **Upload PDF files**
2. **Extract text** using PyPDF
3. **Split content into chunks** for efficient processing
4. **Generate embeddings** using Ollama
5. **Store vectors** in FAISS for fast retrieval
6. **Query the data** using:
   - 📖 **Full Read (MapReduce)** — comprehensive analysis
   - 🎯 **Targeted Retrieval (MMR)** — fast and precise results


### 🌐 Web Workflow
1. Enter a URL
2. Extract content using WebBaseLoader
3. Chunk and embed
4. Store in FAISS
5. Query using selected mode


## 🧩 Core Concepts
- Retrieval-Augmented Generation (RAG)
- FAISS vector similarity search
- Maximum Marginal Relevance (MMR)
- MapReduce-based summarization
- Hierarchical reduction for large documents
- Smart sampling (relevance + coverage)


## 🤝 Contributing

Contributions are welcome!

  1. Fork the repository
  2. Create a feature branch
  3. Commit changes
  4. Open a Pull Request

## 📜 License

This project is open-source and available under the MIT License.

## 💡 Author

Built with focus on performance + accuracy in RAG systems.
