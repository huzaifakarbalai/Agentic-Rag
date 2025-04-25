from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config.config import EMBEDDING_MODEL, NUM_RETRIEVAL_DOCS, ERROR_MESSAGES


class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create a vector store from documents.

        Args:
            documents: List of document chunks

        Raises:
            ValueError: If there's an error creating the vector store
        """
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            raise ValueError(ERROR_MESSAGES["vector_store_error"].format(error=str(e)))

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
