
import numpy as np
import json
from flask import Flask, request, jsonify
import os
import time
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama


app=Flask(__name__)

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf_embedding_model=HuggingFaceBgeEmbeddings(model_name=model_name,
                                            model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

load_saved_db=FAISS.load_local("car_repair_guide",hf_embedding_model,allow_dangerous_deserialization=True)


retriever=load_saved_db.as_retriever(search_kwargs={"k":5})

Loaded_llm_model = Ollama(base_url="http://localhost:11434",model="phi3")

def get_cotext_from_query(retriever,input_query):
    return retriever.invoke(input_query)

def llm_query(retrieved_context,input_quetion,llm_llama_model):
    prompt_template = f"""Final Prompt: Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, strictly don't add anything from your side.

    Context: {retrieved_context}
    Question: {input_quetion}

    Only return the helpful answer. give direct answer with reference from the context
    answer:
    """
    # input_text_to_llm={"text_corpus":prompt_template}
    return llm_llama_model.invoke(prompt_template)


@app.route('/get_answer/', methods=['POST'])
def get_answer():
    review= request.get_json()
    context_from_user_query=get_cotext_from_query(retriever,review["text"])
    output_from_llm_model=llm_query(context_from_user_query,review["text"],Loaded_llm_model)
    
    return jsonify(output_from_llm_model)

if __name__ =="__main__":
    app.run(debug=True, port=4000)