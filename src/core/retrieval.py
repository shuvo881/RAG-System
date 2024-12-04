from langchain_core.tools import tool
from ..pydentic_models.rag_model import State
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

    @tool(response_format="content_and_artifact")
    def retrieve_context(self, query: str):
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
