import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.memory.buffer import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()

# -------------- here i loaded the environment things
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ==================== LLM and Embeddings ============
llm = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2")

# VECTOR_DIR = "kishlay_vectorestore"
VECTOR_DIR = r"C:\Users\kishl\OneDrive\Desktop\GEN\PERSONAL_CHATBOT\kishlay_vectorestore"
VECTOR_PATH = os.path.join(VECTOR_DIR,"index.faiss")


def get_vectorstore():
  """loaded the faiss if its already there"""
  if os.path.exists(VECTOR_PATH):
    vector_db = FAISS.load_local(VECTOR_DIR,embeddings,allow_dangerous_deserialization=True)
    return vector_db
  else:
    print("creating new vector_store.. please wait")

    text_loader = TextLoader("personal.txt",encoding='utf-8')
    json_loader = JSONLoader("personal.json",jq_schema=".[]",text_content=False)
    pdf_loader = PyPDFLoader("kishlay_chatbot_making.pdf")

    text_loaded = text_loader.load()
    json_loaded = json_loader.load()
    pdf_loaded = pdf_loader.load()

    all_data = text_loaded + json_loaded + pdf_loaded


     # ----------------- Splitter appling ----------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_data)

    #  ---------------- create and save faiss index ----------
    vector_db = FAISS.from_documents(split_docs,embeddings)
    vector_db.save_local(VECTOR_DIR)

    print('vectorstore created')

    return vector_db

vector_db = get_vectorstore()

prompt = ChatPromptTemplate.from_template("""
You are 'Kishlay AI' — a friendly, confident, and professional AI version of Kishlay Kumar.

Your role:
- Speak naturally, like Kishlay explaining his own work.
- Always answer in a conversational, human tone — never like a report or documentation.
- Keep responses concise (about 4–5 sentences maximum).
- Never use tables, markdown tables, or structured columns.
- Use simple paragraphs or short bullet points if needed.
- Focus on clarity, natural flow, and friendly explanations.
- When asked about projects or skills, summarize briefly (purpose, tools, and what was learned).
- Do not list unnecessary technical details unless explicitly asked.
- If the user asks about multiple things, list them clearly in bullet or sentence form — never as a table.
- If you are not sure, reply with: "I'm not sure about that yet, but Kishlay can tell you more!"

<context>
{context}
</context>

User Question: {input}
""")



memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

retriever = vector_db.as_retriever()

documents_chain = create_stuff_documents_chain(llm,prompt)
chain = create_retrieval_chain(retriever,documents_chain)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vector_db.as_retriever(search_kwargs={"k": 3})
# )

# retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k":3})

input_text = st.text_input("enter anything you want to ask about kishlay")
submit = st.button("Submit")

if submit:
  if input_text.strip() == "":
    st.warning("enter questions to ask")

  else:
    response = chain.invoke({"input":input_text},)
    if "answer" in response:
      st.write(response["answer"])
      print(response['answer'])
    else:
      st.write(response["result"])
      print(response['result'])



# response = qa_chain.invoke("what is your college name?")
# print(response['result'])