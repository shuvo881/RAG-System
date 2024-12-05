from langchain_core.tools import tool
from .indexing import VectorStoreManager


class DocumentRetrievalService:


    def __init__(self):

        try:
            self.vector_store_manager = VectorStoreManager()
            self.vector_store = self.vector_store_manager._initialize_vector_store()
            if not self.vector_store:
                raise RuntimeError("Failed to initialize the vector store.")
        except Exception as e:
            raise RuntimeError(f"Error initializing DocumentRetrievalService: {e}")

    def retrieve(self, query: str):
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=3)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
