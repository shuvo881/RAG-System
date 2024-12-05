from langgraph.graph import START, StateGraph, MessagesState
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from .retrieval import DocumentRetrievalService 
from .generation import QuestionAnsweringService
from .analyze_query import AnalyzeQueryService


class QuestionAnsweringPipeline:

    def __init__(self,):
        self.graph_builder = StateGraph(MessagesState)
        self._build_pipeline()
        self.graph = self.graph_builder.compile()


    def _build_pipeline(self):
        retrieve = DocumentRetrievalService().retrieve
        generate = QuestionAnsweringService().generate
        query_or_respond = AnalyzeQueryService().query_or_respond
        self.graph_builder.add_node(query_or_respond)
        tools = ToolNode([retrieve])
        self.graph_builder.add_node(tools)
        self.graph_builder.add_node(generate)
        
        self.graph_builder.set_entry_point("query_or_respond")
        self.graph_builder.add_conditional_edges(
            'query_or_respond',
            tools_condition,
            {END: END, "tools": "tools"},
        )
        self.graph_builder.add_edge("tools", "generate")
        self.graph_builder.add_edge("generate", END)


    def stream_responses(self, input_question: str):
        print("Streaming responses...")
        for step in self.graph.stream(
            {"messages": [{"role": "user", "content": input_question}]},
            stream_mode="values",
        ):
            yield step["messages"][-1].pretty_print()



