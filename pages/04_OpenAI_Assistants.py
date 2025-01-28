import streamlit as st
import time
from openai import OpenAI

# Streamlit Sidebar for OpenAI API Key
st.sidebar.header("API Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

# Initialize OpenAI Client
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=api_key)

client = st.session_state.client

# Helper function to create an assistant
def create_assistant():
    if "assistant" not in st.session_state:
        st.session_state.assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="You are a research assistant. Use tools to search the web, scrape content, and save results.",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_wikipedia",
                        "description": "Search Wikipedia for a given query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search term."}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_duckduckgo",
                        "description": "Search DuckDuckGo for a given query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search term."}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "scrape_website_content",
                        "description": "Scrape and extract text content from a given URL.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "The URL to scrape."}
                            },
                            "required": ["url"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "save_to_file",
                        "description": "Save the research content to a .txt file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The content to save."}
                            },
                            "required": ["content"]
                        }
                    }
                }
            ],
            model="gpt-4o"
        )
    return st.session_state.assistant

# Cancel active runs
def cancel_active_runs(thread_id):
    active_runs = client.beta.threads.runs.list(thread_id=thread_id)
    for run in active_runs:
        if run.status in ["in_progress", "queued"]:
            client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            st.write(f"Cancelled active run: {run.id}")

# Polling function to monitor run status
def poll_run_status(run_id, thread_id, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        st.write(f"Run status: {run.status}")
        if run.status in ["completed", "failed"]:
            return run
        elif run.status == "requires_action":
            handle_required_action(run)
        time.sleep(2)
    raise TimeoutError("Run did not complete within the timeout period.")

# Submit tool outputs for required actions
def handle_required_action(run):
    required_action = run.required_action
    if required_action and required_action.type == "submit_tool_outputs":
        tool_calls = required_action.submit_tool_outputs.tool_calls
        tool_outputs = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name == "search_wikipedia":
                tool_outputs.append({"tool_call_id": tool_call.id, "output": "Wikipedia result for query."})
            elif tool_name == "search_duckduckgo":
                tool_outputs.append({"tool_call_id": tool_call.id, "output": "DuckDuckGo result for query."})
            elif tool_name == "scrape_website_content":
                tool_outputs.append({"tool_call_id": tool_call.id, "output": "Scraped website content."})
            elif tool_name == "save_to_file":
                tool_outputs.append({"tool_call_id": tool_call.id, "output": "File saved successfully."})

        client.beta.threads.runs.submit_tool_outputs(
            thread_id=run.thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
        st.write("Submitted tool outputs.")

# Streamlit UI
st.title("OpenAI Assistants API Research App")
query = st.text_input("Enter your query:")

if st.button("Run"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                # Step 1: Create Assistant
                assistant = create_assistant()

                # Step 2: Create a Thread with initial message
                if "thread" not in st.session_state:
                    st.session_state.thread = client.beta.threads.create(
                        messages=[
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    )

                thread = st.session_state.thread
                st.write("Thread Created with Initial Message:", thread)

                # Step 3: Cancel active runs
                cancel_active_runs(thread.id)

                # Step 4: Create a new run
                run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )

                # Step 5: Poll for run status
                run = poll_run_status(run.id, thread.id)

                # Retrieve Messages from the Thread
                messages = client.beta.threads.messages.list(thread_id=thread.id)

                # Extract the assistant's response
                response_message = None
                for message in messages.data:
                    if message.role == "assistant":
                        response_message = message.content
                        break

                if response_message is None:
                    raise ValueError("No assistant message found in the thread.")

                # Display Results
                st.success("Query processed successfully!")
                st.text_area("Results:", response_message, height=200)

            except TimeoutError as e:
                st.error(f"Timeout: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Add GitHub link to sidebar
st.sidebar.markdown(
    "[https://github.com/mc-rodrigo22/langchain_study_2/blob/main/pages/04_OpenAI_Assistants.py)"
)
