# 🧠 Kishlay — AI Personal Chatbot (Built with Streamlit + LangChain)
###   🤖 A friendly, intelligent AI chatbot that represents Kishlay Kumar — built with LangChain, Groq LLM, FAISS, and Streamlit.


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


## 🧩 Tech Stack

**Component**   	   **Technology**

- Frontend -----------        UI	Streamlit

- Backend Logic  ---------- 	Python + LangChain

- LLM Model	Groq ---------   (OpenAI GPT-OSS-120B)

- Embeddings	   -----------   HuggingFace MiniLM L3 v2

- Vector Store	 ----------    FAISS

- Document Loaders	--------- TXT · JSON · PDF

- Environment Mgmt  ---------	 dotenv (.env for API keys)

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
