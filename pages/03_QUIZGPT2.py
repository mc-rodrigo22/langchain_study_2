import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import random
import time

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QUIZGPT",
    page_icon="✔",
)

st.title("QUIZGPT")

with st.sidebar:
    st.header("Settings")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
    
    difficulty = st.selectbox("Select Difficulty Level", ["Easy", "Medium", "Hard"])

    st.markdown(
        "[View on GitHub](https://github.com/mc-rodrigo22/langchain_study_2/blob/main/pages/02_QUIZGPT.py)", unsafe_allow_html=True
    )

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

llm = ChatOpenAI(
    temperature=0.3 if difficulty == "Easy" else 0.7 if difficulty == "Medium" else 1.0,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key=openai_api_key,
).bind(
    function_call={
        "name": "get_questions",
    },
    functions=[function],
)

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

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": format_docs} | prompt | llm
    response = chain.invoke(_docs)
    arguments = json.loads(response.additional_kwargs["function_call"]["arguments"])
    for index in range(len(arguments["questions"])):
        random.shuffle(arguments["questions"][index]["answers"])
    return arguments

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    show_answer = st.checkbox("Display the correct answer when the user selects an incorrect option", False)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    form_key = f"questions_form_{int(time.time() * 1000)}"

    with st.form(form_key):  
        correct_count = 0

        for idx, question in enumerate(response["questions"]):
            value = st.radio(
                f"{idx+1}: {question['question']}",
                [
                    f"{index+1}: {answer['answer']}"
                    for index, answer in enumerate(question["answers"])
                ],
                index=None,
                key=f"question_{idx}",
            )
            isCorrect = False
            if value:
                isCorrect = {"answer": value[3:], "correct": True} in question["answers"]
            if isCorrect:
                correct_count += 1
                st.success("✅ Correct!")
            elif value:
                if show_answer:
                    for index, answer in enumerate(question["answers"]):
                        if "correct" in answer and answer["correct"]:
                            answer_number = index + 1
                            break
                    st.error(f"❌ It's wrong. (# of Answer: {answer_number})")
                else:
                    st.error("❌ It's wrong.")
            st.divider()

        button = st.form_submit_button("Submit")
        if button:
            if correct_count < len(response["questions"]):
                st.warning(f"Your score: {correct_count}/{len(response['questions'])}. Try again!")
                st.experimental_rerun()
            else:
                st.balloons()
                st.success("Perfect Score! Well done!")



                
#250109 코드 에러 아예 없게 & 전체 테스트 해보기 & 깃헙 링크 사이드바 맨 아래로 & 전체 코드 이해하기
