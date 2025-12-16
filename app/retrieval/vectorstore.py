""" VectorStore - Store  and search vectors
    Whe use Chroma for its simplicity and automatic persistence
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from app.config import settings
from pydantic import SecretStr

class VectorStore:
    """ Wrapper around Chroma to handle vector store
        Whats do Chroma?
        - Store embeddings (vectors )
        - Permit search over embeddings
        - Automatic persist data to disk
    """

    def __init__(self):
        """Initialize vector store with Chroma and Azure OpenAI Embeddings"""
        # Create embeddings of Azure
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=SecretStr(settings.AZURE_OPENAI_KEY),
            model=settings.AZURE_OPENAI_EMBED_MODEL
        )

        # Path to persist vector DB
        self.persist_directory = settings.VECTORDB_DIR

        # Vector Store (upload from disk if exist)
        self.vectorstore = None


    def create_from_documents(self, documents: List[Document]) -> None:
        """ Create a new collection with the documents provided
            IMPORTANT: thie overwrite the existing collection in disk

            Args:
                documents: List of Langchain Documents to ingest
        """

        print(f"Creating vector store with {len(documents)} chunks...")

        #Chroma automatically:
        # 1.- Extract the text from the documents
        # 2.- Create embeddings using self.embeddings
        # 3.- Store vectors and metadata 
        # 4.- Persist data to disk
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="rag_collection"
        )

        print(f"Vector store created in {self.persist_directory}.")
        print(f"Total of chunks indexed: {len(documents)}")

    def load_existing(self) -> bool:
        """ Load a vector store from disk if exist
        
            Returns:
                True if load successfully, False otherwise
        """
        try: 
            if not os.path.exists(self.persist_directory):
                print("No existing vector store found.")
                return False
            
            #Load from disk
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )

            #Verify if have data
            count = self.vectorstore._collection.count()
            if count == 0:
                print("Vector store is empty.")
                return False
            print(f"Loaded existing vector store with {count} chunks.")
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """ Add new documents to existing collection
            Args:
                documents: List of Langchain Documents to add
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Load or create a vector store first.")
        
        print(f" Adding {len(documents)} new chunks to vector store...")
        self.vectorstore.add_documents(documents)
        print("New chunks added successfully.")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """ Perform a similarity search in the vector store

            Intern process:
            1. query -> embedding
            2. search in vector DB for top k similar vectors
            3. return the corresponding documents

            Args:
                query: User question to search
                k: Number of top similar chunks to retrieve

            Returns:
                List of Langchain Documents similar to the query
        """
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Load or create a vector store first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """ Perform a similarity search in the vector store with scores
            Returns: 
                List of Tuples (Document, score) similar to the query
                Score: lower is better
        """

        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Load or create a vector store first.")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
