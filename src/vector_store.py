from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config.config import EMBEDDING_MODEL, NUM_RETRIEVAL_DOCS, ERROR_MESSAGES
import os
from pathlib import Path


class VectorStore:
    def __init__(
        self, use_local_storage: bool = True, storage_path: str = "vector_store"
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.use_local_storage = use_local_storage
        self.storage_path = storage_path
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents. If use_local_storage is True and a saved version exists,
        it will load the existing vector store instead of creating a new one.

        Args:
            documents: List of document chunks

        Raises:
            ValueError: If there's an error creating the vector store
        """
        try:
            if self.use_local_storage and self._vector_store_exists():
                print("used local embeddings")
                self.vector_store = self._load_vector_store()
            else:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                if self.use_local_storage:
                    self._save_vector_store()
        except Exception as e:
            raise ValueError(ERROR_MESSAGES["vector_store_error"].format(error=str(e)))

    def _vector_store_exists(self) -> bool:
        """Check if a saved vector store exists."""
        return os.path.exists(os.path.join(self.storage_path, "index.faiss"))

    def _save_vector_store(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store is not None:
            self.vector_store.save_local(self.storage_path)

    def _load_vector_store(self) -> FAISS:
        """Load the vector store from disk."""
        return FAISS.load_local(
            self.storage_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def similarity_search(self, query: str) -> List[Document]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query

        Returns:
            List of relevant documents

        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store first."
            )

        return self.vector_store.similarity_search(query, k=NUM_RETRIEVAL_DOCS)

    def get_context(self, query: str) -> str:
        """
        Get context from vector store for a query.

        Args:
            query: Search query

        Returns:
            Combined context from relevant documents
        """
        docs = self.similarity_search(query)
        return " ".join([doc.page_content for doc in docs])
