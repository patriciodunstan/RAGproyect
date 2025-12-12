import os
from typing import List

from pypdf import PdfReader


class DocumentLoader:


    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Lee un documento PDF  y retorna todo el texto extraído."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        #Limpieza básica
        text = text.replace("\n", " ").replace("\r", " ").strip()

        return text
    
    @staticmethod
    def load_txt(file_path:str) -> str:
        """Lee un archivo de texto simple"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        text = text.replace("\n", " ").replace("\r", " ").strip()

        return text