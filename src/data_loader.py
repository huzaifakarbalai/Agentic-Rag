import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, ERROR_MESSAGES


class DataLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load and split a PDF file into chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of document chunks

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF file is invalid
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                ERROR_MESSAGES["file_not_found"].format(file_path=pdf_path)
            )

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            raise ValueError(
                ERROR_MESSAGES["invalid_pdf"].format(file_path=pdf_path)
            ) from e

    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents based on file extension.

        Args:
            file_path: Path to the document file

        Returns:
            List of document chunks
        """
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            return self.load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
