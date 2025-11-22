# 💬 Kishlay AI — Personal Chatbot

Meet **Kishlay AI**, a personalized chatbot that speaks just like *Kishlay Kumar*!  
It understands his background, projects, skills, and experiences — giving natural, friendly, and context-aware responses.

🚀 **Live App:** [Click to Open on Streamlit](https://your-streamlit-link-here)

---

## 🧠 Features

- 🗣️ **Conversational AI:** Answers naturally like Kishlay Kumar  
- 🔍 **RAG (Retrieval-Augmented Generation):** Uses personal data (text, JSON, PDF)  
- ⚙️ **FAISS Vector Store:** Enables fast and semantic retrieval  
- 🧩 **Groq LLM Integration:** Uses `ChatGroq` for fast and efficient inference  
- 🔤 **Hugging Face Embeddings:** Encodes knowledge base into embeddings  
- 🧾 **Streamlit UI:** Clean, interactive chat interface  
- 🌀 **Spinner Effect:** “Kishlay is thinking...” animation for cool UX  

---

## 🏗️ Tech Stack

| Category | Tools Used |
|-----------|-------------|
| Framework | Streamlit |
| LLM | Groq (`ChatGroq`) |
| Embeddings | Hugging Face Sentence Transformers |
| Vector Store | FAISS |
| Data Sources | `.txt`, `.json`, `.pdf` |
| Chain Logic | LangChain (Retrieval Chain + Prompt Templates) |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/kishlay-ai-chatbot.git
cd kishlay-ai-chatbot

```


## 📜 Overview

- Kishlay AI is a personalized chatbot that knows everything about my skills, projects, and background.

- It uses Retrieval-Augmented Generation (RAG) — combining large-language-model reasoning with document-based knowledge to answer queries naturally and accurately.

- The chatbot is deployed via Streamlit for a clean and interactive UI and powered by a FAISS vector store for fast document retrieval.
  

## ⚙️ Features

- ✔ Conversational Personality — Speaks like Kishlay Kumar, friendly and professional.

- ✔ RAG Pipeline — Retrieves answers directly from my documents (PDF, JSON, TXT).

- ✔ LangChain Integration — Uses modern LangChain chains (create_retrieval_chain, create_stuff_documents_chain).

- ✔ Groq LLM (OSS-120B) — Super-fast inference via the Groq API.

- ✔ HuggingFace Embeddings — “sentence-transformers/paraphrase-MiniLM-L3-v2” for vectorization.

- ✔ Streamlit UI — Interactive web app for easy Q&A.

- ✔ Prompt Control — Enforces natural, human-like tone (no tables, short 4–5 line replies).

- ✔ Local Vector Persistence — FAISS index saved for instant reloads.



## 📁 Project Structure
``` bash
Kishlay_AI_Chatbot/
│
├── fresh_chatbot.py          # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .env                      # API keys (Groq, HuggingFace)
│
├── kishlay_vectorestore/     # Saved FAISS index
│   └── index.faiss
│
├── personal.txt              # Text data (bio, skills)
├── personal.json             # Structured info (projects, achievements)
├── kishlay_chatbot_making.pdf # Portfolio / resume data
└── README.md
```

## 🚀 How to Run Locally
#### 1️⃣ Clone the repository
``` bash
git clone https://github.com/<your-username>/Kishlay-AI-Chatbot.git
cd Kishlay-AI-Chatbot
```

#### 2️⃣ Create and activate a virtual environment
``` bash
python -m venv env
env\Scripts\activate   # On Windows
source env/bin/activate  # On macOS/Linu
```


#### 3️⃣ Install dependencies
``` bash
pip install -r requirements.txt
```

#### 4️⃣ Set up your .env file

Create a .env in the project root:

``` bash
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

#### 5️⃣ Run the Streamlit app

``` bash
streamlit run fresh_chatbot.py
```

✅ Open the browser at → http://localhost:8501
