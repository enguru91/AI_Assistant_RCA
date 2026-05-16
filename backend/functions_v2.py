# Author: Eshan Gurusinghe
# Email: engurusinghe91@gmail.com
# Version: 2nd version:GUI base working script
# Date: 2025/07/07
import os
import requests
import pickle
import glob
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
import numpy as np
import hashlib
import sentence_transformers
import openai
import streamlit as st
from dotenv import load_dotenv

# Load environment variables and initialize constants
load_dotenv()
# Initialize parameters
api_key = os.environ["LLM_API_KEY"]
api_url = os.environ["LLM_BASE_URL"]
api_embed_model = os.environ["MODEL_NAME_LLM"]
reranker_model = "eli-reranker-small-1".strip()
INDEX_NAME = "*"
resources_path = "./NRJ-Dev/Include/ProcedureFiles/"# Resources location
embeddings_path = "./NRJ-Dev/Include/Outputs/Embeddings/"# Embedding data location
output_dir = "./NRJ-Dev/Include/Outputs/Chats/"# chat outputs location
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")# Get current date and time
checksum_file = "current_checksum_" + current_datetime # Define new filename prefix for checksum
filename_prefix = "ResourcesFile_" + current_datetime # Define new filename prefix for resources
current_checksum_value = None # Initialize flag and checksum variable
# Ensure the output directories exists
os.makedirs(resources_path, exist_ok=True)
os.makedirs(embeddings_path, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# API key check function
def check_api_key_first():
    if os.path.exists(api_key_location):
        with open(api_key_location, 'r') as f:
            lines = f.read().splitlines()
        if len(lines) == 1 and lines[0].startswith('eli-'):
            print("Valid .eli-key file detected. Key prefix is valid")
            api_key = open(api_key_location).read().strip()
            return api_key
        else:
            print("Invalid .eli-key file detected! Key prefix is invalid")
            return ""
    else:
        return False
    
# API key add/modify function
def add_modify_api_key (new_key:str):
    # Update the API key
    with open(api_key_location, 'w') as f:
        f.write(new_key)
    print(f"New API key added! {new_key}")
    return True

# Delete function to delete remaining resource files in resources_path
def delete_remaining_resources():
    if os.path.isdir(resources_path):
        for filename in os.listdir(resources_path): 
            if filename.startswith('ResourcesFile_') and filename.endswith('.txt'):
                file_path = os.path.join(resources_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted resource file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete resource file {file_path}: {e}")
    else:
        print(f"resources file does not exist or is not a directory: {resources_path}")
    
    if os.path.isdir(resources_path):
        for filename in os.listdir(resources_path): 
            if filename.startswith('current_checksum_') and filename.endswith('.txt'):
                file_path = os.path.join(resources_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted checksum file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete checksum file {file_path}: {e}")
    else:
        print(f"checksum file does not exist or is not a directory: {resources_path}")

    if os.path.isdir(embeddings_path):
        for filename in os.listdir(embeddings_path): 
            if filename.startswith('current_embeddings_') and filename.endswith('.pkl'):
                file_path = os.path.join(embeddings_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted embeddings file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete embeddings file {file_path}: {e}")
    else:
        print(f"embedding file does not exist or is not a directory: {embeddings_path}")

#check uploaded files from the GUI
def check_for_uploaded_files(uploaded_files):
    """
    Handle the uploaded files: save them, compute checksum, and decide whether to recreate embeddings.
    """
    current_checksum_path = check_exists_checksum_file(resources_path)
    if uploaded_files:
        # Save uploaded files and create new checksum
        print("Uploading files detected.")
        if current_checksum_path:
            try:
                with open(current_checksum_path, 'r') as file:
                    # Assuming the checksum is stored as a single line in the file
                    current_checksum_value = file.read().strip()
                    print(f"Current checksum value: {current_checksum_value}")
                    new_checksum_value = save_uploaded_files(uploaded_files)
                    print(f"New checksum after saving files: {new_checksum_value}")
            except FileNotFoundError:
                print(f"The checksum file {current_checksum_path} does not exist.")
                current_checksum_value = None
            except Exception as e:
                print(f"An error occurred while reading the checksum file: {e}")
                current_checksum_value = None
        else:
            print("No current checksum file available")
            current_checksum_value = None

        # check for checksum differences
        if current_checksum_value ==  new_checksum_value:
            print("Resources checksum matches, No resource files changed! Loading existing embeddings.")
            # resources_changed = False
            return load_embeddings_from_file()
        elif current_checksum_value !=  new_checksum_value:
            print("Resources checksum does not match! Creating new embeddings.")
            embeddings_data = create_embeddings()
            # resources_changed = True
            # Update stored checksum file with current checksum
            with open(current_checksum_path, 'w') as f:
                f.write(new_checksum_value)
            print(f"New checksum {new_checksum_value} rewrited!")
            return embeddings_data
        elif not current_checksum_path:
            # No existing checksum file            
            print("No checksum file exists! Creating new embeddings and checksum.")
            new_checksum_value = save_uploaded_files(uploaded_files)
            # Update stored checksum file with current checksum
            with open(current_checksum_path, 'w') as f:
                f.write(new_checksum_value)
            print(f"New checksum {new_checksum_value} rewrited!")
            embeddings_data = create_embeddings()
            # resources_changed = True
            return embeddings_data
    else:
        print("No uploaded files detected.")

# Function to save uploaded files
def save_uploaded_files(uploaded_files):
    """
    Save uploaded text files to the resources directory after deleting existing ones.
    """
    # Delete existing files starting with "ResourcesFile_"
    pattern = os.path.join(resources_path, "ResourcesFile_*")
    files_to_delete = glob.glob(pattern)
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted resource file: {file}")
        except Exception as e:
            print(f"Error deleting resource file {file}: {e}")

    # Save each uploaded file into resources_path
    for index, uploaded_file in enumerate(uploaded_files): 
        new_filename = f"{filename_prefix}_{index + 1}.txt"
        file_path = os.path.join(resources_path, new_filename)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        print(f"Saved uploaded file: {file_path}")

    # Compute and save checksum
    current_checksum_value = compute_resources_checksum(resources_path)
    # checksum_path = os.path.join(resources_path, checksum_file + ".txt")
    # with open(checksum_path, 'w') as f:
    #     f.write(current_checksum_value)
    # print(f"Checksum after saving files: {current_checksum_value}")
    # return current_checksum_value
    if current_checksum_value:
        try:
            for filename in sorted(os.listdir(resources_path)):
                if filename.endswith('.txt') and filename.startswith('current_checksum_'):
                    filepath = os.path.join(resources_path, filename)
                    print (filepath)
                    try:
                        with open(filepath, 'w') as f:
                            f.write(current_checksum_value)
                        print(f"Checksum written to {filepath}: {current_checksum_value}")
                    except Exception as e:
                        print(f"Error writing checksum to {filepath}: {e}")
        except Exception as e:
            print(f"Error processing files in {resources_path}: {e}")
        return current_checksum_value
    if not current_checksum_value:
         print("An error occured related to checksum file creation")    

# Function to compute checksum of uploaded files after saving uploaded files
def compute_resources_checksum(directory):
    """
    Compute MD5 checksum of all resource files to detect changes.
    """
    hash_md5 = hashlib.md5()
    # Read only files starting with "ResourcesFile_" for checksum
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt') and filename.startswith('ResourcesFile_'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Find existing checksum files
def check_exists_checksum_file(resources_path):
    try:
        # List all files that start with 'current_checksum_'
        matching_files = [f for f in os.listdir(resources_path) if f.startswith('current_checksum_')]
        if matching_files:
            matching_files.sort()  # Assuming chronological order
            current_checksum_path = os.path.join(resources_path, matching_files[0])
            return current_checksum_path
        else:
            print("No checksum files found. Create new one executing")
            new_checksum_filename = f"current_checksum_{current_datetime}.txt"
            new_checksum_filepath = os.path.join(resources_path, new_checksum_filename)
            try:
                with open(new_checksum_filepath, 'w') as f:
                    f.write(current_checksum_value)
                print(f"Created new checksum file {new_checksum_filepath} with checksum: {current_checksum_value}")
            except Exception as e:
                print(f"Error creating new checksum file {new_checksum_filepath}: {e}")
            return new_checksum_filepath
    except FileNotFoundError:
        print(f"The directory {resources_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while searching for checksum files: {e}")
        return None

#create embedding from text files
def create_embeddings(embeddings_dir: str = embeddings_path):
    """
    Create embeddings from the resources and save them.
    """
    # Delete existing embedding files starting with "embeddings_"
    pattern = os.path.join(embeddings_dir, "current_embeddings_*")
    files_to_delete = glob.glob(pattern)
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted embedding file: {file}")
        except Exception as e:
            print(f"Error deleting embedding file {file}: {e}")

    # Create new embeddings
    print("Creating new embeddings...")
    texts = load_resources(resources_path)
    embeddings = eli_encode(api_embed_model, texts)

    if embeddings != {}:
        # Save embeddings file with timestamp
        new_filename = f"current_embeddings_{current_datetime}.pkl"
        new_file_path = os.path.join(embeddings_dir, new_filename)
        with open(new_file_path, 'wb') as f:
            pickle.dump({'texts': texts, 'vectors': embeddings['vectors']}, f)
        print(f"Created new embeddings file at {new_filename}.")
        return texts, np.array(embeddings['vectors'])
    elif not embeddings.get('vectors'):
        print("Failed to create embeddings. Aborting embedding creation.")
        return [], np.array([])    

# Existing load_embeddings function
@st.cache_data
def load_embeddings_from_file(embeddings_dir: str = embeddings_path):
    """
    Load embeddings from the latest pickle file.
    """
    # Check for existing embedding files
    print("Checking for existing embedding files...")
    pkl_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.pkl') and f.startswith('current_embeddings_')]
    if pkl_files:
        # Load the latest embedding file
        pkl_files.sort()  # Assuming chronological order
        embeddings_file = os.path.join(embeddings_dir, pkl_files[-1])
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded precomputed embeddings from {embeddings_file}.")
        return data['texts'], np.array(data['vectors'])
    else:
        # If no embeddings found, create new ones
        print("No embeddings found. Creating new embeddings.")
        return create_embeddings(embeddings_dir)
        
#loading multipal text files
def load_resources(directory : str):
    """
    Load text resource files from the specified directory.
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('ResourcesFile_'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    print(f"Loaded {len(texts)} texts from resources.")
    return texts

#embbeding the text data
@st.cache_data
def eli_encode(model: str, texts: list):
    """
    Make API call to ELI text embedding endpoint and returns vectors response.
    """
    try:
        payload = {
            "model": model,
            "texts": texts,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{api_url}/api/v1/bi_encoder/encode",
            json=payload,
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        if response.ok:
            return response.json()
        print("Failed to get response.", response.text)
    except Exception as ex:
        print(f"An error occurred during embedding: {ex}")
        return {}

#get the all the library categories    
def eli_semantic_categories(index:str = INDEX_NAME):
    """Make API call to list the data source categories in ELI Semantic."""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{api_url}/api/v1/rag/categories/{index}",
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        if response.ok:
            return response.json().get("categories", [])
        print("Failed to get response.", response.text)
    except Exception as ex:
        print(f"An error occurred: {ex}.")

#get the all the library IDs    
def eli_semantic_cpi_library_identities(index:str = INDEX_NAME):
    """Make API call to list the cpi library identities in ELI Semantic."""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{api_url}/api/v1/rag/cpi/library_identities/{index}",
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        if response.ok:
            return response.json().get("categories", [])
        print("Failed to get response.", response.text)
    except Exception as ex:
        print(f"An error occurred: {ex}.")

categories = eli_semantic_categories(index=INDEX_NAME)
cpi_library_identities = eli_semantic_cpi_library_identities(index=INDEX_NAME)

#filter search with Intelligent Search    
def eli_semantic_rag (
    query: str,
    index: str = INDEX_NAME,
    chat_history: list = [],
    model: str = "",
    category: str = categories,
    cpi_library_id: str = cpi_library_identities,
    rerank: bool = True,
    top_k: int = 10,
    generate_answer: bool = True
):
    """
    Make API call to ELI Semantic endpoint and ask questions on Ericsson internal documents.

    query: the question to ask
    category: the category of information source, empty to cover all
    """
    
    try:
        payload = {
            "index_name": index,
            "model": model,
            "query": query,
            "chat_history": chat_history,
            "category": category,
            "cpi_library_id": cpi_library_id,
            "rerank": rerank,
            "top_k": top_k,
            "generate_answer": generate_answer,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{api_url}/api/v1/rag/query",
            json=payload,
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        if response.ok:
            return response.json()
        print("Failed to get response.", response.text)
    except Exception as ex:
        print(f"An error occurred: {ex}.")

def get_most_relevant_docs(query, texts):
    """
    Retrieve the most relevant documents based on the query.
    """
    text_pairs = [(query, doc) for doc in texts]
    #run ELI document reranker method
    Reranked_response = eli_score_rank(model = reranker_model, text_pairs = text_pairs)
    # return eli_score_rank(model = reranker_model, text_pairs = texts, query = query)

    # Process Response
    if Reranked_response:
        print("Model:", Reranked_response.get("model"))  # the actual model used
        scores = Reranked_response.get("scores", [])
        print("Scores:", scores)
        # Rerank documents based on scores
        reranked_documents = [
            texts[idx] for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ]
        print("Reranked Documents")
        return reranked_documents
        # for num, doc in enumerate(reranked_documents, 1):
        #     print(f"{num}. {doc}")
    else:
        return []
    
# ELI rerank sample_documents based on their relevance to the query
def eli_score_rank(model: str, text_pairs: list):
    """
    Make API call to ELI text ranking endpoint and returns scores and reranked list of texts.

    text_pairs: list of pairs of texts
    """
    try:
        payload = {
            "model": model,
            "text_pairs": text_pairs,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{api_url}/api/v1/cross_encoder/score",
            json=payload,
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        if response.ok:
            print("Reranking success!\n\n", response.json())
            return response.json()
        print("Failed to get response.", response.text)
    except Exception as ex:
        print(f"An error occurred: {ex}")

# Store chat history as an output
def generate_output_file(chat_history):
    """
    Generates a text file with the chat history.
    """
    if not chat_history:
        print("No chat history to write.")
        return None
    filename = f"chat_output_{current_datetime}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as file:
        for entry in chat_history:
            role = entry.get("role", "unknown").capitalize()
            content = entry.get("content", "").strip()
            file.write(f"{role}: {content}\n\n")    
    print(f"Chat history successfully written to {filepath}")
    return filepath

#ELI chat
def eli_chat(
        messages=[], 
        model:str = "",
        max_new_tokens: int = 4096,
        temperature: float =0.3, 
        top_p: float =0.8,
        top_k: int =10,
    ):
    """
    Make API call to ELI LLM Chat endpoints.
    """

    # Chat model token adjustment
    if model == "mistral-12b":
        print(f"inquire LLM with {model} model")
        max_new_tokens = 8192
    elif model == "phi4-14b":
        print(f"inquire LLM with {model} model")
        max_new_tokens = 8192
    elif model == "qwen2.5-7b":
        print(f"inquire LLM with {model} model")
        max_new_tokens = 8192
    elif model == "llama3.1-8b":
        print(f"inquire LLM with {model} model")
        max_new_tokens = 8192
    elif model == "deepseeker1-14b":
        print(f"inquire LLM with {model} model")
        max_new_tokens = 8192

    try:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{api_url}/api/v1/llm/chat",
            json=payload,
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        if response.ok:
            print("Received response from LLM.")
            return response.json()
        print(f"LLM chat failed with status code {response.status_code}: {response.text}")
        return "Failed to get response."
    except Exception as ex:
        print(f"An error occurred during LLM chat: {ex}")
        return f"An error occurred: {ex}"