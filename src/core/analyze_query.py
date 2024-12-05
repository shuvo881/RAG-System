from langgraph.graph import MessagesState
from .model_emb import Loader
from .retrieval import DocumentRetrievalService



class AnalyzeQueryService:
    
    def __init__(self):
        try:
            self.loader = Loader()
            self.llm = self.loader.load_model()
            if not self.llm:
                raise RuntimeError("Failed to load the language model.")
        except Exception as e:
            raise RuntimeError(f"Error initializing QuestionAnsweringService: {e}")

    def query_or_respond(self, state: MessagesState):
        try:
            dr = DocumentRetrievalService()
            retrieve = dr.retrieve
            llm_with_tools = self.llm.bind_tools([retrieve])
            response = llm_with_tools.invoke(state["messages"])

            if not response:
                raise RuntimeError("Failed to generate a query.")
            
            return {"messages": [response]}

        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {e}")


