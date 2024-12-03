from langchain_core.prompts import ChatPromptTemplate
from src.pydentic_models.rag_model import State, Search
from .model_emb import Loader


class AnalyzeQueryService:
    
    def __init__(self):
        try:
            self.loader = Loader()
            self.llm = self.loader.load_model()
            if not self.llm:
                raise RuntimeError("Failed to load the language model.")
        except Exception as e:
            raise RuntimeError(f"Error initializing QuestionAnsweringService: {e}")

    def analyze_query(self, state: State):
        try:
            structured_llm = self.llm.with_structured_output(Search)
            query = structured_llm.invoke(state["question"])

            if not query:
                raise RuntimeError("Failed to generate a query.")
            
            return {"query": query}

        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {e}")


