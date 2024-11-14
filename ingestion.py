import os

from dotenv import (load_dotenv)
load_dotenv()

from consts import INDEX_NAME
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from firecrawl import FirecrawlApp
from langchain.schema import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
def ingest_docs():
    loader = ReadTheDocsLoader("./langchain-docs/api.python.langchain.com/en/latest/", encoding='UTF-8')
    raw_documents = loader.load()
    print(f"loaded{len(raw_documents)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"loaded {len(documents)} documents")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )

def ingest_docs2() -> None:
    app = FirecrawlApp(api_key= os.environ['FIRECRAWL_API_KEY'])
    url = "https://humanidades.com/acero/#:~:text=Seg%C3%BAn%20el%20m%C3%A9todo%20utilizado%20para%20darle%20forma%3A%201,de%20l%C3%A1minas%20m%C3%A1s%20o%20menos%20gruesas%20y%20planas."

    page_content = app.scrape_url(url=url,
                                  params= {
                                      "onlyMainContent":True
                                  })
    print(page_content)
    doc=Document(page_content=str(page_content), metadata={"source":url})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents([doc])

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name = "proyect1-index"
    )


if __name__ == "__main__":
    ingest_docs2()
