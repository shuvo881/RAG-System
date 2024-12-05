from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from .model_emb import Loader


class QuestionAnsweringService:
    
    def __init__(self):
        try:
            self.loader = Loader()
            self.llm = self.loader.load_model()
            if not self.llm:
                raise RuntimeError("Failed to load the language model.")
        except Exception as e:
            raise RuntimeError(f"Error initializing QuestionAnsweringService: {e}")

    def generate(self, state: MessagesState):
        try:

            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]

            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                f"{docs_content}"
            )
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages

            # Run
            response = self.llm.invoke(prompt)
            return {"messages": [response]}

        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {e}")


