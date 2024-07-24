import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain.chains import RetrievalQA

def main():
    # Load environment variables
    load_dotenv()
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
    os.environ['QDRANT_API_KEY'] = os.getenv("QDRANT_API_KEY")

    # Initialize the language model
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.5, model_kwargs={'max_length': 8192}, max_new_tokens=4096
    )


    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Define the prompt template
    template = """
    You are a direct and concise assistant. Answer the question using only the information provided in the context. Give only the specific answer requested, with no additional explanation or information.
    Reply with - I cannot answer that question with my limitations.
    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Initialize Qdrant vector store
    url = "https://3bb301e0-87bb-460b-9ef0-c79b4c1b53e4.us-east4-0.gcp.cloud.qdrant.io:6333"
    apikey = os.getenv("QDRANT_API_KEY")
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name="assignrag",
        url=url,
        api_key=apikey
    )

    # Set up the retriever and compressor
    retriever = qdrant.as_retriever(search_kwargs={"k": 3})
    compressor = JinaRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # Initialize the QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever, chain_type_kwargs={"prompt": prompt})

    # Streamlit UI
    st.title("FAQ Assistant on Real Estate Ventures")

    # List of project names
    project_names = [
        "Maple Residences",
        "Oakwood Apartments",
        "Pine Valley Estates",
        "Cedar Heights",
        "Birchwoods Residences",
        "Willow Park",
        "Aspen Place",
        "Elmwood Gardens",
        "Redwood Avenue",
        "Sycamore Towers",
        "Cypress Courts",
        "Maplewood Homes",
        "Hemlock Heights",
        "Cedar Springs",
        "Redwood Ridge"
    ]

    # Displaying the list of project names in columns
    
    st.write("The list of all the Apartments are as shown below :")

    # Number of columns you want to display
    num_columns = 3

    # Creating columns
    columns = st.columns(num_columns)

    # Distributing project names into the columns
    for idx, project in enumerate(project_names):
        col = columns[idx % num_columns]
        col.write(project)

    st.write("There are 3 types of flats - 2BHK, 3BHK & 4BHK at all of the above apartments, feel free to ask any questions about it in the below box")
    st.markdown("""Sample questions such as : \n\n 1. What is type of kitchen available at Redwood Ridge ? \n\n 2. What is the cost of 3 BHK flat at Aspen Place ? \n\n 3. How many units are available in Maple Residences ?
    
    
    
    """)

    query = st.text_input("Enter your query:")
    if query:
        response = qa.invoke(query)
        st.write("Answer:", response['result'])

if __name__ == '__main__':
    main()
