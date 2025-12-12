"""
    Loeader - Upload documents to vector convertsion
    suported formats: PDF, TXT, DOCX
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2TxtLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    Uploard document of different formats and convert to Langchain Document

    Document have:
    - page_content: the text content of the document
    - metadata: dictionary with metadata information(source, page, etc)
    """

    SUPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2TxtLoader
    }

    @classmethod
    def load_file(cls, file_path: Path) -> List[Document]:
        """Upload document and return list of documents
        why return List? because one PDF can have multiple pages to generate multiple documents

        Args:
            file_path:  Route to the file

        Returns:
            List of documents (everyone withe page_content and metadata)
        """

        extension = file_path.suffix.lower()
        if extension not in cls.SUPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {extension}\n"
                             f"Suported formats: {list(cls.SUPORTED_EXTENSIONS.keys())}"
                        )
        
        # Select the appropriate loader based on file extension
        loader_class = cls.SUPORTED_EXTENSIONS[extension]
        loader = loader_class(str(file_path))

        #load documents can be multiple (PDF with multiple pages)
        documents = loader.load()

        # Add metada information for tracking
        for doc in documents:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = extension
            # if not have page, add indicator 
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0
        
        return documents
    
    @classmethod
    def load_directory(cls, directory: Path) -> List[Document]:
        """Upload all documents supported in a directory
        
        Args:
            directory: Route to the directory

        Returns:
            List of documents (everyone withe page_content and metadata)
        """

        all_documents = []
        files_processed = 0

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPORTED_EXTENSIONS:
                try:
                    docs = cls.load_file(file_path)
                    all_documents.extend(docs)
                    files_processed += 1
                    print(f" {file_path.name}: {len(docs)} páginas/secciones")
                except Exception as e:
                    print(f"Error in {file_path.name}: {str(e)}")

        print(f"\n Total: {files_processed} archivos, {len(all_documents)} documents")
        return all_documents
