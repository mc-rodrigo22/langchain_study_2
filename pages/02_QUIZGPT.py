import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain_core.runnables import RunnableMap
import os

# Sidebar: User-provided API Key
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please provide your OpenAI API key to proceed.")
        st.stop()

    # Check if the API key is valid
    if not openai_api_key.startswith("sk-"):
        st.error("Invalid OpenAI API key. Please enter a valid key.")
        st.stop()

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = openai_api_key

# Difficulty adjustment function
def adjust_difficulty(difficulty):
    if difficulty == "Easy":
        return 0.3
    elif difficulty == "Medium":
        return 0.5
    else:  # Hard
        return 0.7

# Define the function for function calling
function = {
    "name": "get_questions",
    "description": "It is a function that requires a questions array consisting of a question and multiple choices.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            },
        },
        "required": ["questions"],
    },
}

# LLM initialization with .bind()
def create_llm(difficulty):
    return ChatOpenAI(
        temperature=adjust_difficulty(difficulty),
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={
            "name": "get_questions",
        },
        functions=[function],
    )

# Prompt template
prompt = PromptTemplate.from_template(
    """
You are a professional question maker who creates questions to test students' knowledge based on the provided documents.
Create 10 questions using the information found in the given context.
Each question must have 4 options, with only one correct answer.
Ensure all questions are concise and unique.

--------Context--------
{context}
-----------------------
"""
)

# Helper functions
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    try:
        # 파일 내용을 읽기
        file_content = file.read()
        file_path = f"./.cache/quiz_files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 파일 로딩 및 텍스트 분할
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        if not docs:
            st.error("The file could not be processed. Please check the file format.")
            return None

        # 텍스트 분할
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        docs = splitter.split_documents(docs)

        return docs

    except Exception as e:
        st.error(f"Error while processing the file: {e}")
        return None

# State management
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "current_source" not in st.session_state:
    st.session_state.current_source = None
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Medium"

# Sidebar inputs
with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        ("File", "Wikipedia Article"),
        key="source_choice"
    )

    difficulty = st.selectbox(
        "Select difficulty level", 
        ["Easy", "Medium", "Hard"], 
        key="difficulty"
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
            key="file_upload"
        )
        if file:
            docs = split_file(file)
            source = f"file:{file.name}"
    else:
        topic = st.text_input("Search Wikipedia...", key="wiki_search")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            docs = retriever.get_relevant_documents(topic)
            source = f"wiki:{topic}"

    st.markdown("---")
    st.markdown("[View this project on GitHub](https://github.com/mc-rodrigo22/langchain_study_2/blob/main/pages/02_QUIZGPT.py)")

# Detect changes and regenerate quiz if needed
if docs and (source != st.session_state.current_source or st.session_state.difficulty != difficulty):
    st.session_state.quiz_data = None
    st.session_state.current_source = source
    llm = create_llm(difficulty)

    # Create a Runnable chain
    chain = RunnableMap({"context": lambda x: format_docs(x)}) | prompt | llm

    try:
        response = chain.invoke(docs)
        st.session_state.quiz_data = response.additional_kwargs["function_call"]["arguments"]
    except Exception as e:
        st.error(f"Error while generating quiz: {e}")

# Display quiz
if st.session_state.quiz_data:
    response = json.loads(st.session_state.quiz_data)
    with st.form("questions_form"):
        score = 0
        for idx, question in enumerate(response["questions"]):
            value = st.radio(
                f"{idx+1}: {question['question']}",
                [f"{index+1}: {answer['answer']}" for index, answer in enumerate(question["answers"])],
                index=None,
            )
            if value:
                selected_answer = int(value.split(":")[0]) - 1
                if question["answers"][selected_answer]["correct"]:
                    score += 1
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect.")
        button = st.form_submit_button("Submit")
        if button:
            if score == len(response["questions"]):
                st.balloons()
                st.success(f"Perfect Score! {score}/{len(response['questions'])}")
            else:
                st.warning(f"Your score: {score}/{len(response['questions'])}. Try again!")
else:
    st.markdown(
        """
        Welcome to QuizGPT.

        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
