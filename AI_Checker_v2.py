# Author: Eshan Gurusinghe
# Email: engurusinghe91@gmail.com
# Version: 2.1 version:GUI base working script 2025/07/07
# 2.2 version:GUI base integrated with internal data libraries 2025/08/01
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import concurrent.futures
import os
import time

# Import backend functions from the 'backend' module
from backend.functions_v2 import (
    eli_chat,
    generate_output_file,
    check_for_uploaded_files,
    load_embeddings_from_file,
    get_most_relevant_docs,
    eli_semantic_rag,
    delete_remaining_resources,
)

# Load environment variables and initialize constants
load_dotenv()
delete_remaining_resources() #delete any remaining resource files 
# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    print(st.session_state['chat_history'])
if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'vectors' not in st.session_state:
    st.session_state['vectors'] = np.array([])
if 'upload_success' not in st.session_state:
    st.session_state['upload_success'] = False
if 'edit_mode' not in st.session_state:
    st.session_state['edit_mode'] = False

# Load or create embeddings at startup
if 'texts' not in st.session_state or 'vectors' not in st.session_state:
    # Load or create embeddings
    texts, vectors = load_embeddings_from_file()
    # vectors = np.array(vectors)
    st.session_state['texts'] = texts
    st.session_state['vectors'] = vectors
# Streamlit GUI layout
st.title("AI Assistant")
st.write("Lets generate your command list with the assistant of AI!")

# if st.session_state.get('upload_status_message'):
#     st.success(st.session_state['upload_status_message'])
#     # Clear the message so it doesn't show again
#     st.session_state['upload_status_message'] = None

# Dropdown for model selection
model_choice = st.selectbox("AI model", options=["mistral-12b","phi4-14b","qwen2.5-7b","llama3.1-8b","deepseekr1-14b"]).strip()

# Text input for user message
user_input = st.text_area("Enter your prompt here:", key='user_input')

# Buttons layout
col_left, col_middle, col_right = st.columns([1, 2, 1])

with col_middle:
    uploaded_files = st.file_uploader(
        "Upload resource files (TXT or CSV, less than 200MB each)",
        type=["txt", "csv"], 
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "default_uploader")
    )
with col_left:
    chat_button = st.button("Ask AI", disabled=(key_status == "" or key_status == False))
    use_internal_libs = st.toggle("Use internal libraries", key="use_internal_libraries")
with col_right:
    no_history = len(st.session_state['chat_history'])
    if no_history == 0: 
        print_button = st.button("Export Chat", disabled=no_history)
    else:
        print_button = st.button("Export Chat")
    new_chat_button = st.button("New Chat")
# with col_right:
#     no_history = len(st.session_state['chat_history']) == 0
#     if st.button("Export Chat", disabled=no_history):
#         save_chat()        
#     new_chat_button = st.button("New Chat")

# Function to call the LLM with a timeout
def get_llm_response_with_timeout(func, timeout_seconds=240, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            # Return None or raise a specific error to be caught later
            return None

def process_user_input(query = user_input, text_data  = st.session_state['texts'], use_internal=False, vectors = st.session_state['vectors']):
    if not query.strip():
        st.warning("Please enter a prompt!")
    else:    
        st.session_state['chat_history'].append({"role": "User", "content": query})
        if use_internal:
            with st.spinner('Searching inside the internal CPI libraries...'):
                print("Calling internal CPI libraries with query")
            print (text_data) # Uploaded document data
            text_data_str = ': '.join(text_data) # convert list to string
            chat_history = [
                {"role": "user", "content" : query},
                {"role": "assistant", "content" : text_data_str},
            ]
            internal_CPI_data = eli_semantic_rag(query = chat_history[0]['content'], chat_history = chat_history, model = model_choice)
            if internal_CPI_data:
                st.info("Internal CPI library references found, further processing your request!")
                print (internal_CPI_data) # references found internal data
                try:
                    # Structure the prompt for the LLM & combine the relevant texts & internal CPI data
                    prompt_messages = [
                        {"role": "system", "content": "Follow the instructions carefully and give accurate answers."},
                        {"role": "user", "content": f"Uploaded content:\n{text_data_str}\nInternal CPI data:\n{internal_CPI_data}\nQuestion: {query}"}
                    ]

                    # Call LLM with timeout
                    response = get_llm_response_with_timeout(eli_chat, messages=prompt_messages, model=model_choice)
                    if response is None:
                        st.error("4 mins LLM timeout occured! rephrase your message & try again!")
                        st.session_state['chat_history'].append({"role": "Assistant", "content": "*Timeout: No response received within 4 mins.*"})
                        # Display chat history
                        st.subheader("Chat History")
                        for entry in st.session_state['chat_history']:
                            role = entry.get("role", "unknown")
                            msg = entry.get("content", "")
                            if role == "User":
                                st.markdown(f"**You:** {msg}")
                            else:
                                st.markdown(f"**Assistant:** {msg}")
                        return
                    elif response and isinstance(response, dict) and "choices" in response:
                        st.success("The AI assistant, answer generated...!")
                        content = response["choices"][0]["message"].get("content", "").strip()
                        st.session_state['chat_history'].append({"role": "Assistant", "content": content}) # Append chat history as dicts
                        # Display chat history
                        st.subheader("Chat History")
                        for entry in st.session_state['chat_history']:
                            role = entry.get("role", "unknown")
                            msg = entry.get("content", "")
                            if role == "User":
                                st.markdown(f"**You:** {msg}")
                            else:
                                st.markdown(f"**Assistant:** {msg}")
                    else:
                        content = f"Failed to get a valid response: {response}"
                        st.warning(content)
                except Exception as e:
                    st.error(f"An error occurred while searching on internal libraries: {e}")
        else:
            try:
                # Retrieve most relevant documents based on the user input
                relevant_texts = get_most_relevant_docs(
                    query = query, 
                    texts = text_data
                )
                if not relevant_texts:
                    st.warning("No relevant internal information found to provide context but further processing!")
                    # Structure the prompt for the LLM
                    prompt_messages = [
                        {"role": "system", "content": "You are a knowledgeable assistant."},
                        {"role": "user", "content": f"Question: {query}"}
                    ]
                    st.info("Generating an AI response now...") 
                    # Call LLM with timeout
                    response = get_llm_response_with_timeout(eli_chat, messages=prompt_messages, model=model_choice, temperature=0.7)
                    if response is None:
                        st.error("4 mins LLM timeout occured! rephrase your message & try again!")
                        st.session_state['chat_history'].append({"role": "Assistant", "content": "*Timeout: No response received within 4 mins.*"})
                        # Display chat history
                        st.subheader("Chat History")
                        for entry in st.session_state['chat_history']:
                            role = entry.get("role", "unknown")
                            msg = entry.get("content", "")
                            if role == "User":
                                st.markdown(f"**You:** {msg}")
                            else:
                                st.markdown(f"**Assistant:** {msg}")
                        return
                    elif response and isinstance(response, dict) and "choices" in response:
                        st.success("The AI assistant, answer generated...!")
                        content = response["choices"][0]["message"].get("content", "").strip()
                        st.session_state['chat_history'].append({"role": "Assistant", "content": content})
                        # Display chat history
                        st.subheader("Chat History")
                        for entry in st.session_state['chat_history']:
                            role = entry.get("role", "unknown")
                            msg = entry.get("content", "")
                            if role == "User":
                                st.markdown(f"**You:** {msg}")
                            else:
                                st.markdown(f"**Assistant:** {msg}")
                    else:
                        content = f"Failed to get a valid response: {response}"
                        st.error(content)
                else:
                    st.info("Generating an AI response now...")               
                    # Combine the relevant texts to form the context
                    relevant_context = "\n\n".join(relevant_texts)
                    print(f"Selected context:\n{relevant_context}")
                
                    # Structure the prompt for the LLM
                    prompt_messages = [
                        {"role": "system", "content": "You are a knowledgeable assistant."},
                        {"role": "user", "content": f"Context:\n{relevant_context}\nQuestion: {query}"}
                    ]

                    # Call LLM with timeout
                    response = get_llm_response_with_timeout(eli_chat, messages=prompt_messages, model=model_choice)
                    if response is None:
                        st.error("4 mins LLM timeout occured! rephrase your message & try again!")
                        st.session_state['chat_history'].append({"role": "Assistant", "content": "*Timeout: No response received within 4 mins.*"})
                        # Display chat history
                        st.subheader("Chat History")
                        for entry in st.session_state['chat_history']:
                            role = entry.get("role", "unknown")
                            msg = entry.get("content", "")
                            if role == "User":
                                st.markdown(f"**You:** {msg}")
                            else:
                                st.markdown(f"**Assistant:** {msg}")
                        return
                    elif response and isinstance(response, dict) and "choices" in response:
                        st.success("The AI assistant, answer generated...!")
                        content = response["choices"][0]["message"].get("content", "").strip()
                        st.session_state['chat_history'].append({"role": "Assistant", "content": content})
                        # Display chat history
                        st.subheader("Chat History")
                        for entry in st.session_state['chat_history']:
                            role = entry.get("role", "unknown")
                            msg = entry.get("content", "")
                            if role == "User":
                                st.markdown(f"**You:** {msg}")
                            else:
                                st.markdown(f"**Assistant:** {msg}")
                    else:
                        content = f"Failed to get a valid response: {response}"
                        st.error(content)                
            except Exception as e:
                st.error(f"An error occurred processing your request: {e}")

# Handle File Uploads
if uploaded_files:
        try:
            with st.spinner('Processing uploaded files...'):
                upload_result = check_for_uploaded_files(uploaded_files)
            if upload_result:
                texts, vectors = load_embeddings_from_file()
                st.session_state['texts'] = texts
                st.session_state['vectors'] = vectors
                st.session_state['upload_success'] = True
        except Exception as e:
            st.error(f"An error occurred while uploading files: {e}")
else:
    st.warning("No files were uploaded")


# Show success message only once
if st.session_state.get('upload_success', False):
    st.success("File upload completed!")


# Handle File Uploads
# if uploaded_files and not st.session_state['upload_success']:
#     try:
#         with st.spinner('Processing uploaded files...'):
#             upload_result = check_for_uploaded_files(uploaded_files)
#         if upload_result:
#             texts, vectors = load_embeddings_from_file()
#             st.session_state['texts'] = texts
#             st.session_state['vectors'] = vectors
#             st.session_state['upload_success'] = True
#             st.rerun()
#     except Exception as e:
#         st.error(f"An error occurred while uploading files: {e}")
# elif st.session_state['upload_success'] and not st.session_state['upload_success_shown']:
#     st.success("File upload completed!")
#     time.sleep(3)
#     st.session_state['upload_success_shown'] = True
#     st.rerun()
# elif st.session_state['upload_success'] and st.session_state['upload_success_shown']:
#     st.session_state['upload_success'] = False
#     st.session_state['upload_success_shown'] = False
# elif not uploaded_files:
#     st.warning("No files were uploaded")




# Handle chat history save
def save_chat():
    try:
        output_path = generate_output_file(st.session_state['chat_history'])
        if output_path:
            st.success(f"Chat history saved at: {output_path}")
            folder_path = os.path.dirname(output_path)
            absolute_folder_path = os.path.abspath(folder_path)
            os.startfile(absolute_folder_path)
        elif st.session_state['chat_history'] == []:
            st.error("Chat history is empty. Nothing to export.")
    except Exception as e:
        st.error(f"Failed to write chat history to a file: {e}")

# Button click handling
if chat_button:
    process_user_input(use_internal=use_internal_libs)
    #st.rerun()  # Rerun to update the chat display immediately after processing
if print_button:
    save_chat()
if new_chat_button:
    st.session_state.clear()
    st.session_state["uploader_key"] = str(time.time())  # Unique key to reset uploader
    st.session_state["user_input"] = ""  # Clear the text input field
    st.session_state["use_internal_libraries"] = False   # Reset toggle
    st.session_state["new_key"] = False   # Reset toggle
    st.info("Chat history & other parameters are cleared!")
    time.sleep(2)
    st.rerun()