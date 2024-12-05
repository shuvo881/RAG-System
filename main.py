from src.core.pipeline import QuestionAnsweringPipeline


if __name__ == "__main__":
    input_message = "Who is CEO of Serenus One?"

    # Initialize the pipeline with the State model
    print("Initializing pipeline...")
    pipeline = QuestionAnsweringPipeline()
    # Stream and print responses from the pipeline
    for response in pipeline.stream_responses(input_message):
        print(response, end="")