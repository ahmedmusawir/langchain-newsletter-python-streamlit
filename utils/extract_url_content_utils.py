from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import nltk

# Ensure nltk resources are available
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def extract_url_content(urls):
    # Setup embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize an empty list to store successfully loaded documents
    successful_docs = []

    # Use UnstructuredURLLoader to load data
    for url in urls:
        try:
            loader = UnstructuredURLLoader([url])
            data = loader.load()
            if data:
                successful_docs.extend(data)
                print(f"Successfully loaded content from {url}")
            else:
                print(f"No content found at {url}")
        except Exception as e:
            print(f"Error fetching or processing {url}, exception:\n{e}")
            continue

    if not successful_docs:
        print("No valid documents loaded. Exiting.")
        return None

    # Splitting the text content
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Load splits into a document
    docs = text_splitter.split_documents(successful_docs)
    print(f"Split into {len(docs)} document chunks")

    if not docs:
        print("No documents to split, check data and splitter parameters")
        return None

    # Vectorize data with FAISS
    db = FAISS.from_documents(docs, embeddings)

    return db
