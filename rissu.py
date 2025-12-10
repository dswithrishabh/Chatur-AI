'''import streamlit as st
import os
import torch
import base64
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# CPU FORCE

torch.device("cpu")


# BACKGROUND IMAGE

def set_background_image(image_path):
	if os.path.exists(image_path):
		with open(image_path, "rb") as f:
			img_bytes = f.read()
		encoded = base64.b64encode(img_bytes).decode()
		st.markdown(
			f"""
			<style>
			.stApp {{
				background-image: url("data:image/png;base64,{encoded}");
				background-size: cover;
				background-position: center;
				background-attachment: fixed;
			}}
			</style>
			""",
			unsafe_allow_html=True
		)

# Convert .avif → .png manually and set here
set_background_image("BGGG.avif")


# CONSTANTS

TEXTS_DIRECTORY = "text_files"
os.makedirs(TEXTS_DIRECTORY, exist_ok=True)

# EMBEDDINGS

embedding_model = HuggingFaceEmbeddings(
	model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# VECTOR STORE

def initialize_vector_store():
	if os.path.exists("faiss_index"):
		return FAISS.load_local(
			"faiss_index",
			embedding_model,
			allow_dangerous_deserialization=True
		)
	else:
		return FAISS.from_texts([""], embedding_model)

vector_store = initialize_vector_store()


# LOAD & CHUNK DOCUMENTS

def load_and_chunk_documents_to_vector_store(file_path):
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	
	if file_path.lower().endswith(".pdf"):
		loader = PyPDFLoader(file_path)
	else:
		loader = TextLoader(file_path)
	
	docs = loader.load()
	if not docs:
		st.warning(f"No content found in {file_path}")
		return
	
	chunks = splitter.split_documents(docs)
	if not chunks:
		st.warning(f"No chunks created from {file_path}")
		return
	
	vector_store.add_documents(chunks)
	vector_store.save_local("faiss_index")


# MODELS

model_dict = {
	"KingNish/Qwen2.5-0.5b-Test-ft": "KingNish/Qwen2.5-0.5b-Test-ft",
	"microsoft/phi-4": "microsoft/phi-4",
	"mistralai/Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407"
}


# SESSION STATE

if "chat_history" not in st.session_state:
	st.session_state.chat_history = []

if "current_topic" not in st.session_state:
	st.session_state.current_topic = "General"

if "topic_changed" not in st.session_state:
	st.session_state.topic_changed = False

if "rating_list" not in st.session_state:
	st.session_state.rating_list = []


# SIDEBAR

new_topic = st.sidebar.text_input("Enter Topic:", st.session_state.current_topic)
if new_topic and new_topic != st.session_state.current_topic:
	st.session_state.current_topic = new_topic
	st.session_state.topic_changed = True

selected_model_name = st.sidebar.selectbox("Select Model", list(model_dict.keys()))
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150, 50)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 1.0, 0.05)

language = st.sidebar.selectbox("Language", ["English", "Hindi"])

# Upload TXT / PDF
uploaded_file = st.sidebar.file_uploader("Upload File (.txt or .pdf)", type=["txt","pdf"])
if uploaded_file:
	file_path = os.path.join(TEXTS_DIRECTORY, uploaded_file.name)
	with open(file_path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	load_and_chunk_documents_to_vector_store(file_path)
	st.sidebar.success("File uploaded and processed!")

# Rating system
st.sidebar.markdown("### Rate the Answer")
rating = st.sidebar.radio("Rate:", ["Excellent ⭐⭐⭐⭐⭐", "Good ⭐⭐⭐", "Poor ⭐"])
if st.sidebar.button("Submit Rating"):
	st.session_state.rating_list.append(rating)
	st.sidebar.success("Thank you for your feedback!")


# LOAD LLM MODEL

@st.cache_resource
def load_model(model_name):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	return tokenizer, model

tokenizer, model = load_model(model_dict[selected_model_name])

generator = pipeline(
	"text-generation",
	model=model,
	tokenizer=tokenizer,
	max_new_tokens=max_tokens,
	temperature=temperature,
	top_p=top_p,
	device="cpu"
)

llm = HuggingFacePipeline(pipeline=generator)


# PROMPT TEMPLATE

prompt_template = """
You are an expert AI tutor.
Use ONLY the context below to answer exactly based on content.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
	template=prompt_template,
	input_variables=["context", "question"]
)


# CUSTOM REPLIES

def custom_reply(user_msg):
	msg = user_msg.lower()

	if "kisne banaya" in msg or "who created you" in msg:
		return "Mujhe **Mr. Rishabh Pandey** aur unki team ne banaya hai 😊"
	if "team member" in msg or "team me kaun" in msg:
		return "Team members: **Mr. Saurabh Pathak**, **Mr. Shivam**, **Mr. Bhavesh** 😎"
	if "good morning" in msg:
		return "🌞 **Good Morning!** Main aapki kaise help kar sakta hoon?"
	if "good evening" in msg:
		return "🌆 **Good Evening!** Main aapki kaise help kar sakta hoon?"
	if "good night" in msg:
		return "🌙 **Good Night!** Dhyan rakhna! Aur kuch help chahiye?"
	if "jai shree ram" in msg:
		return "🙏 **Jai Shree Ram!**"
	if "har har mahadev" in msg:
		return "🕉️ **Har Har Mahadev!**"
	if "hod" in msg:
		return "Himalaya College Lucknow CSE Department ke HOD: **Mr. Jeetendra Vishwakarma**"
	if "director" in msg:
		return "Himalaya College ke Director: **Mr. Preetesh Saxena**"
	if "dean" in msg:
		return "Himalaya College ke Dean: **Mr. Ashish Dixit**"
	if "executive director" in msg:
		return "Executive Director: **Mrs. Samta Bafila**"
	if "course" in msg or "courses" in msg:
		return "Available Courses:\n- B.Tech\n- Polytechnic\n- B.Pharma\n- D.Pharma\n- BBA\n- MBA\n- ITI"
	if "college ke bare me" in msg or "about college" in msg:
		return (
			"Himalaya College Lucknow ek top emerging institute hai, jaha modern labs, "
			"experienced faculty aur student-friendly environment milta hai. "
			"Yaha placements aur practical training dono par strong focus diya jata hai. "
			"Aapki career growth ke liye yeh ek perfect choice hai! 🎓🔥"
		)
	if "address" in msg or "kaha hai" in msg or "location" in msg:
		return "📍 **Lucknow, Bakshi Ka Talab, Near Banke Nagar Chauraha**"
	return None


# KEYWORD SEARCH

def keyword_search(query, docs):
	return [d for d in docs if query.lower() in d.page_content.lower()]


# RAG PIPELINE
 
def rag_pipeline(query):
	docs = vector_store.similarity_search(query, k=5)
	exact_docs = keyword_search(query, docs)
	if exact_docs:
		docs = exact_docs

	context = "\n".join([d.page_content for d in docs])
	final_prompt = prompt.format(context=context, question=query)
	result = llm.generate([final_prompt])
	raw = result.generations[0][0].text
	if "Answer:" in raw:
		return raw.split("Answer:")[1].strip()
	return raw.strip()

#
# UI HEADER
#
col1, col2 = st.columns([1, 6])
with col1:
	st.image("chatur baba.jpeg", width=90)
with col2:
	st.markdown(
		"""
		<h2 style="
			color:#4B0082;
			font-family: 'Georgia', serif;
			font-style: italic;
			text-shadow: 3px 3px 5px #FFB6C1;
			margin: 0;
		">Chatur AI — AskBabaChatur</h2>
		""",
		unsafe_allow_html=True
	)
st.markdown("<h4 style='color:#4B0082; font-family:Georgia; font-style:italic;'>What can I help with?</h4>", unsafe_allow_html=True)


# USER INPUT

user_input = st.text_input("Ask your question", placeholder="Ask a question...", label_visibility="hidden")

if user_input:
	# 1️⃣ Check custom replies first
	answer = custom_reply(user_input)
	# 2️⃣ If no custom reply, run RAG
	if not answer:
		answer = rag_pipeline(user_input)
	
	st.session_state.chat_history.append({"user": user_input, "bot": answer})
	st.write(f"**You:** {user_input}")
	st.write(f"**AI:** {answer}")


# CHAT HISTORY and CLEAR button

st.markdown("### Chat History")
if st.button("🧹 Clear Chat", key="clear_btn"):
	st.session_state.chat_history = []
	st.rerun()

for chat in st.session_state.chat_history:
	st.write(f"**You:** {chat['user']}")
	st.write(f"**AI:** {chat['bot']}")'''


import streamlit as st
import os
import torch
import base64
import difflib
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ================================
# FORCE CPU
# ================================
torch.device("cpu")

# ================================
# BACKGROUND IMAGE
# ================================
def set_background_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_background_image("BGGG.avif")

# ================================
# CONSTANTS
# ================================
TEXTS_DIRECTORY = "text_files"
os.makedirs(TEXTS_DIRECTORY, exist_ok=True)

# ================================
# EMBEDDINGS
# ================================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================================
# VECTOR STORE INIT
# ================================
def initialize_vector_store():
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        return FAISS.from_texts([""], embedding_model)

vector_store = initialize_vector_store()

# ================================
# LOAD & CHUNK DOCUMENTS
# ================================
def load_and_chunk_documents_to_vector_store(file_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()
    if not docs:
        st.warning(f"No content found in {file_path}")
        return

    chunks = splitter.split_documents(docs)
    if not chunks:
        st.warning(f"No chunks created from {file_path}")
        return

    vector_store.add_documents(chunks)
    vector_store.save_local("faiss_index")

# ================================
# MODELS
# ================================
model_dict = {
    "KingNish/Qwen2.5-0.5b-Test-ft": "KingNish/Qwen2.5-0.5b-Test-ft",
    "microsoft/phi-4": "microsoft/phi-4",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407"
}

# ================================
# SESSION STATE
# ================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_topic" not in st.session_state:
    st.session_state.current_topic = "General"

if "topic_changed" not in st.session_state:
    st.session_state.topic_changed = False

if "rating_list" not in st.session_state:
    st.session_state.rating_list = []

# ================================
# SIDEBAR
# ================================
# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.title("⚙  chatur baba ki Settings")
    st.link_button("📎 Open My Linkdin Profile", "https://www.linkedin.com/in/rishabh-pandey-9a241b364?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app")
    st.link_button("📎 Open My Profile","https://github.com/dswithrishabh/Chatur-AI")

new_topic = st.sidebar.text_input("Enter Topic:", st.session_state.current_topic)
if new_topic and new_topic != st.session_state.current_topic:
    st.session_state.current_topic = new_topic
    st.session_state.topic_changed = True

ai_speed = st.toggle("⚡ Fast Mode")

selected_model_name = st.sidebar.selectbox("Select Model", list(model_dict.keys()))
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150, 50)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 1.0, 0.05)

language = st.sidebar.selectbox("Language", ["Auto", "English", "Hindi"])

rating = st.sidebar.radio("Rate:", ["Excellent ⭐⭐⭐⭐⭐", "Good ⭐⭐⭐", "Poor ⭐"])
if st.sidebar.button("Submit Rating"):
    st.session_state.rating_list.append(rating)
    st.sidebar.success("Thank you for your feedback!")

# ================================
# LOAD LLM MODEL
# ================================
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model(model_dict[selected_model_name])

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    device="cpu"
)

llm = HuggingFacePipeline(pipeline=generator)

# ================================
# LANGUAGE DETECTOR
# ================================
def detect_language(text):
    hindi_chars = sum(1 for ch in text if '\u0900' <= ch <= '\u097F')

    if hindi_chars >= 2:
        return "hindi"

    hindi_keywords = ["kya", "kaise", "kaun", "kaha", "hai", "tha", "thi", "hun", "ho", "rha", "rhi"]
    for w in hindi_keywords:
        if w in text.lower():
            return "hindi"

    return "english"

# ================================
# FUZZY MATCH HELPER
# ================================
def contains(word_list, msg):
    msg_words = msg.lower().split()
    for w in word_list:
        for mw in msg_words:
            if difflib.SequenceMatcher(None, w, mw).ratio() > 0.75:
                return True
    return False

# ================================
# CUSTOM REPLIES (Auto Hindi-English)
# ================================
def custom_reply(user_msg):
    msg = user_msg.lower()
    lang = detect_language(msg)

    # CREATOR
    if contains(["kisne", "banaya", "created", "creator","who created you"], msg):
        return "Mujhe *Mr. Rishabh Pandey* ne banaya hai 😊" if lang=="hindi" else "I was created by *Mr.Rishabh Pandey* 😊"

    # TEAM
    if contains(["team", "member", "members","who is your team members"], msg):
        return "Team members: *Mr. Saurabh Pathak, **Mr. Shivam, **Mr. Bhavesh* 😎" if lang=="hindi" else "Team members are: *Mr. Saurabh Pathak, Mr. Shivam, Mr. Bhavesh* 😎"

    # GREETINGS
    if "good morning" in msg:
        return "🌞 सुप्रभात! मैं आपकी कैसे मदद कर सकता हूँ?" if lang=="hindi" else "🌞 Good Morning! How can I help you today?"
    if "good evening" in msg:
        return "🌆 शुभ संध्या! कैसे हो?" if lang=="hindi" else "🌆 Good Evening! How can I help you?"
    if "good night" in msg:
        return "🌙 शुभ रात्रि! ध्यान रखना!" if lang=="hindi" else "🌙 Good Night! Take care!"

    # RELIGIOUS
    if contains(["jai", "ram"], msg):
        return "🙏 जय श्री राम!" if lang=="hindi" else "🙏 Jai Shree Ram!"
    if contains(["mahadev"], msg):
        return "🕉️ हर हर महादेव!" if lang=="hindi" else "🕉️ Har Har Mahadev!"

    # COLLEGE DETAILS
    if contains([" cse hod"," who is the hod of cse branch"], msg):
        return "CSE विभाग के HOD: *Mr. Jeetendra Vishwakarma" if lang=="hindi" else "CSE Department HOD: **Mr. Jeetendra Vishwakarma*"
    if contains([" electrical hod","who is the hod of electrical branch"], msg):
        return "EC विभाग के HOD: *Mr. Sandeep Verma" if lang=="hindi" else "EC Department HOD: **Mr. Sandeep Verma*"
    if contains([" electronics hod","who is the hod of electronic branch"], msg):
        return "electronic विभाग के HOD: *Mrs. Kiran *" if lang=="hindi" else "Electronic Department HOD: *Mrs. Kiran*"

    if contains(["director","who is the director of himalayan college"], msg):
        return "Director: *Mr. Preetesh Saxena" if lang=="hindi" else "Director: **Mr. Preetesh Saxena*"
    if contains(["dean","who is the dean of himalayan college"], msg):
        return "Dean: *Mr. Ashish Dixit" if lang=="hindi" else "Dean: **Mr. Ashish Dixit*"
    if contains(["executive", "director","who is the executive director in himalayan","what is the name of himalayan college executive director director"], msg):
        return "Executive Director: *Mrs. Samta Bafila" if lang=="hindi" else "Executive Director: **Mrs. Samta Bafila*"
    if contains(["course", "courses","how many branches are there in himalayan college","himalayan me kitni branch chati hai"], msg):
        return (
            "उपलब्ध कोर्स:\n- B.Tech\n- Polytechnic\n- B.Pharma\n- D.Pharma\n- BBA\n- MBA\n- ITI"
            if lang=="hindi"
            else "Available Courses:\n- B.Tech\n- Polytechnic\n- B.Pharma\n- D.Pharma\n- BBA\n- MBA\n- ITI"
        )
    if contains(["college", "about"], msg):
        return (
            "Himalaya College Lucknow एक emerging institute है जहाँ practical training और placements पर ज़ोर दिया जाता है।"
            if lang=="hindi"
            else "Himalaya College Lucknow is an emerging institute focused on placements and practical training."
        )
    if contains(["address", "location", "kaha"], msg):
        return "📍 लखनऊ, बख्शी का तालाब, बांके नगर चौराहा" if lang=="hindi" else "📍 Lucknow, Bakshi Ka Talab, Near Banke Nagar Chauraha"
    if contains(["email", "mail", "contact", "phone", "number", "mobile","himalayan college ka contact number do","himalayan college ka mail  do","himalayan college email","himalayan college phone number","himalayan college contact number"], msg):
        if lang == "hindi":
            return (
                "📞 संपर्क जानकारी:\n"
                "Phone: +91 7897000465\n7311111465\n"
                "📧 Email: hitmlkoedu465@gmail.com"
            )
        else:
            return (
                "📞 Contact Details:\n"
                "Phone: +91 7897000465\n7311111465\n"
                "📧 Email: hitmlkoedu465@gmail.com"
            )
    # UNIVERSITY NAME QUERY
    if contains([
        "university", 
        "college ka university", 
        "university name", 
        "affiliation", 
        "affiliated",
        "college kis university se hai","himalayan college is affiliated to which university"
    ], msg):
        if lang == "hindi":
            return "🏛 हमारा कॉलेज *Dr.ABJ ABDUL KALAM TECHNICAL UNIVERSITY * से affiliated है।"
        else:
            return "🏛 Our college is affiliated with *Dr.ABJ ABDUL KALAM TECHNICAL UNIVERSITY*."

    return None

# ================================
# FUZZY RAG SEARCH
# ================================
def keyword_search(query, docs):
    matched = []
    for d in docs:
        score = difflib.SequenceMatcher(None, query.lower(), d.page_content.lower()).ratio()
        if score > 0.45:
            matched.append(d)
    return matched

# ================================
# RAG PIPELINE (Hindi + English auto)
# ================================
def rag_pipeline(query):
    lang = detect_language(query)

    docs = vector_store.similarity_search(query, k=5)

    fuzzy_docs = keyword_search(query, docs)
    if fuzzy_docs:
        docs = fuzzy_docs

    context = "\n".join([d.page_content for d in docs])

    final_prompt = f"""
You are a multilingual AI.

User language: {lang}

If language is hindi → Answer fully in Hindi.
If language is english → Answer fully in English.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    result = llm.generate([final_prompt])
    raw = result.generations[0][0].text

    if "Answer:" in raw:
        return raw.split("Answer:")[1].strip()
    return raw.strip()

# ================================
# UI HEADER
# ================================
col1, col2 = st.columns([1, 6])
with col1:
    st.image("chatur baba.jpeg", width=90)
with col2:
    st.markdown(
        """
        <h2 style="
            color:#4B0082;
            font-family: 'Georgia', serif;
            font-style: italic;
            text-shadow: 3px 3px 5px #FFB6C1;
            margin: 0;
        ">Chatur AI — AskBabaChatur</h2>
        """,
        unsafe_allow_html=True
    )
st.markdown("<h4 style='color:#4B0082; font-family:Georgia; font-style:italic;'>What can I help with?</h4>", unsafe_allow_html=True)

# ================================
# USER INPUT
# ================================
user_input = st.text_input("Ask your question", placeholder="Ask a question...", label_visibility="hidden")

if user_input:
    answer = custom_reply(user_input)
    if not answer:
        answer = rag_pipeline(user_input)

    st.session_state.chat_history.append({"user": user_input, "bot": answer})
    st.write(f"*You:* {user_input}")
    st.write(f"*AI:* {answer}")

# ================================
# CHAT HISTORY
# ================================
st.markdown("### Chat History")
if st.button("🧹 Clear Chat", key="clear_btn"):
    st.session_state.chat_history = []
    st.rerun()

for chat in st.session_state.chat_history:
    st.write(f"*You:* {chat['user']}")
    st.write(f"*AI:* {chat['bot']}")