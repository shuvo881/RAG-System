from src.core.pipeline import QuestionAnsweringPipeline
from src.pydentic_models.rag_model import State


if __name__ == "__main__":
    # input_message = "How many microservices are there in the project? and what are their names?"
    input_message = "What is the name of the project?"
    # Initialize the pipeline with the State model
    pipeline = QuestionAnsweringPipeline(State)
    # Stream and print responses from the pipeline
    for response in pipeline.stream_responses(input_message):
        print(response, end="")