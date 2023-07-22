import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from chromadb import Client
import boto3
import json
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import time

# Download the Punkt tokenizer
nltk.download('punkt')

# Initialize clients
chroma_client = Client()
endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b-f"

# SentenceTransformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
@st.cache_resource
def get_embeddings(text):
  return model.encode(sent_tokenize(text)).tolist() 

# Upload page  
def upload_page(collection, file_id, page_num, page_text):
  if page_text.strip():  # Check if the page text is not empty
    embeddings = get_embeddings(page_text)
    if embeddings:  # Check if embeddings were generated
      page_embedding = np.mean(embeddings, axis=0).tolist()
      collection.add(
        documents=[page_text],
        metadatas=[{"file_id": file_id, "page": page_num}],
        ids=[f"{file_id}_page{page_num}"],
        embeddings=[page_embedding]
      )
    else:
      print(f"No embeddings generated for page {page_num} in file {file_id}.")
  else:
    print(f"Page {page_num} in file {file_id} is empty or contains only images.")


# Query endpoint
def query_endpoint(payload):
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

# Streamlit UI and app flow
def main():
  st.title('Just A Rather Very Intelligent System (J.A.R.V.I.S)')
  # Check if the collection exists, if not create it
  if 'pdf_pages' not in chroma_client.list_collections():
    try:
      chroma_client.create_collection('pdf_pages')
    except ValueError as e:
      if "already exists" in str(e):
        pass  # Collection already exists, no need to create it
      else:
        raise  # Re-raise any other ValueError

  collection = chroma_client.get_collection('pdf_pages')

  uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
  if uploaded_file is not None:
    file_id = str(time.time())  # Unique identifier for each file
    with pdfplumber.open(uploaded_file) as pdf:
      for i, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        upload_page(collection, file_id, i+1, page_text)

  query = st.text_input("Enter your query")
  if st.button("Submit Query"):
    # Query ChromaDB for relevant documents
    embeddings = get_embeddings(query)
    results = collection.query(query_texts=[query], n_results=1)
    
    if results['documents']:
        top_result_text = results['documents'][0][0]

        # Use the top result text as input to the model
        dialog = [
            {"role": "system", "content": "You are chatting with an AI assistant."},
            {"role": "user", "content": top_result_text}
        ]
        payload = {
            "inputs": [dialog], 
            "parameters": {"max_new_tokens": 500, "top_p": 0.9, "temperature": 0.6}
        }
        result = query_endpoint(payload)[0]
        st.write(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        st.write("\n==================================\n")
    else:
        st.write("No matching documents found for the query.")

if __name__ == "__main__":
    main()