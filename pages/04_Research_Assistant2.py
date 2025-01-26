import streamlit as st
from openai import OpenAI
import time
from typing import List

# ============================================================
# Streamlit interface setup.

st.set_page_config(page_title="OpenAI Assistant", layout="wide")

# Sidebar setup for API key input and GitHub link.
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
st.sidebar.markdown("[GitHub Repository](https://github.com/mc-rodrigo22/langchain_study_2/blob/main/pages/04_Research_Assistant2.py)")

# Ensure API key is provided.
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

# ============================================================
# Define Function Calling tools.

def duckduckgo_search(query: str) -> str:
    return f"Simulated search results for '{query}' from DuckDuckGo."

def wikipedia_search(query: str) -> str:
    return f"Simulated Wikipedia summary for '{query}'."

def scrape_web_content(url: str) -> str:
    return f"Scraped content from URL: {url}"

def save_to_txt(text: str) -> str:
    with open("research_results.txt", "w") as file:
        file.write(text)
    return "Research results saved to 'research_results.txt'."

# ============================================================
# Create Assistant with tools.
assistant = client.beta.assistants.create(
    name="Research Assistant",
    # 모델이 엉뚱한 툴을 막 호출하지 않도록 추가 지시
    instructions=(
        "You are a knowledgeable research assistant. "
        "You ONLY have the following tools, with EXACT parameter names:\n"
        "1) duckduckgo_search(query)\n"
        "2) wikipedia_search(query)\n"
        "3) scrape_web_content(url)\n"
        "4) save_to_txt(text)\n\n"
        "Never call any other function. "
        "Only call each tool once per user query if needed. "
        "After calling a tool, finalize your answer using the tool result. "
        "Do not repeatedly call the same tool or attempt to call unsupported tools."
    ),
    model="gpt-4o",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "duckduckgo_search",
                "description": "Search the web using DuckDuckGo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wikipedia_search",
                "description": "Search Wikipedia for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scrape_web_content",
                "description": "Scrape content from a given URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape."
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_to_txt",
                "description": "Save given text to a local txt file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to save."
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    ]
)

# ============================================================
# Initialize session state.

if "thread" not in st.session_state:
    st.session_state.thread = client.beta.threads.create()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 세션에 이전 호출 기록 저장용 딕셔너리
if "last_tool_calls" not in st.session_state:
    st.session_state.last_tool_calls = {}

# ============================================================
# Handle requires_action state.

def handle_requires_action(run_id: str, thread_id: str):
    run_details = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    required_action = getattr(run_details, "required_action", None)

    st.write("DEBUG: run_details =", run_details)
    st.write("DEBUG: required_action =", required_action)

    # required_action이 객체 형태이므로, submit_tool_outputs가 None이 아닌지 확인
    if required_action and required_action.submit_tool_outputs is not None:
        tool_calls = required_action.submit_tool_outputs.tool_calls
        if not tool_calls:
            st.write("DEBUG: tool_calls is empty; nothing to submit.")
            # 필요하다면 empty라도 제출 가능
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=[]
            )
            return

        # 이제부터 tool_calls 처리
        tool_outputs = []
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args_str = tool_call.function.arguments  
            # fn_args_str는 JSON 문자열이므로, dict로 로드 필요
            import json
            fn_args = json.loads(fn_args_str)

            # ... 이하 동일 로직
            result = None
            if fn_name == "wikipedia_search":
                result = wikipedia_search(fn_args.get("query", ""))
            elif fn_name == "duckduckgo_search":
                result = duckduckgo_search(fn_args.get("query", ""))
            elif fn_name == "scrape_web_content":
                result = scrape_web_content(fn_args.get("url", ""))
            elif fn_name == "save_to_txt":
                result = save_to_txt(fn_args.get("text", ""))

            st.write(f"DEBUG: Tool call = {fn_name}, args = {fn_args}, result = {result}")

            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": result if result else "No result"
            })

        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
        )

    else:
        st.write("DEBUG: required_action.submit_tool_outputs is None. Doing nothing.")


# ============================================================
# Main UI for conversation.

st.title("OpenAI Assistant")
st.write("Enter your query below and let the assistant perform research for you.")

def handle_query(query):
    thread_id = st.session_state.thread.id
    max_wait_time = 120
    max_attempts = 10
    attempts = 0
    start_time = time.time()

    # 사용자 메시지 추가
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=query
    )

    # Run 실행
    st.info("Starting query execution...")
    run_response = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant.id
    )
    run_id = run_response.id

    # Run 상태 확인
    while True:
        elapsed_time = time.time() - start_time
        run_details = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        run_status = run_details.status

        if run_status == "completed":
            st.success("Query completed successfully!")
            break
        elif run_status == "requires_action":
            if attempts >= max_attempts:
                st.error("Exceeded maximum attempts to handle requires_action.")
                return
            attempts += 1
            handle_requires_action(run_id, thread_id)
        elif run_status in {"failed", "cancelled"}:
            st.error(f"Run ended: {run_status}. Please try again.")
            return
        elif elapsed_time > max_wait_time:
            st.error("Run timeout. The operation took too long.")
            return

        st.info(f"Current status: {run_status} ({elapsed_time:.1f} seconds elapsed)")
        time.sleep(2)

    # Thread에서 마지막 메시지 가져오기
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages_list = list(messages)

    if messages_list:
        latest_message = messages_list[-1]
        if hasattr(latest_message, 'content'):
            content = latest_message.content
            if isinstance(content, list):
                # content가 list 형태면 block 단위로 텍스트를 추출
                message_text = " ".join(
                    block.text.value for block in content if hasattr(block, 'text')
                )
            else:
                message_text = str(content)
        else:
            message_text = "No content available in the message."

        st.session_state.conversation.append({"role": "user", "content": query})
        st.session_state.conversation.append({"role": "assistant", "content": message_text})
    else:
        st.error("No messages found in the thread.")

query = st.text_input("Your query:")
if st.button("Run Query"):
    if query:
        # reset last_tool_calls for each new query if desired
        st.session_state.last_tool_calls = {}
        handle_query(query)
    else:
        st.warning("Please enter a query.")

# Display conversation history.
st.subheader("Conversation History")
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"**User:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Assistant:** {message['content']}")
