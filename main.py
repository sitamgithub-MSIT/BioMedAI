# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import logging
import gradio as gr
import requests


def describe_image(image, question):
    """
    Generate a response for the given image and question using the running server.

    Args:
        - image: Image to describe.
        - question: Question to ask the model.

    Returns:
        Generated response from the model.
    """
    try:
        # Save the image to a temporary file
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Create the payload for the request
        payload = {"image_path": image_path, "question": question}

        # Send the request to the server and get the response
        url = "http://localhost:8000/predict"
        response = requests.post(url, json=payload)

        # Check the response status code
        if response.status_code != 200:
            return f"Server returned status code {response.status_code}"

        # Get the response data and return the generated response
        response_data = response.json()
        logging.info("Response generated successfully.")
        return response_data["response"]

    # Handle exceptions
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Image, text query for the input
image = gr.Image(type="pil", label="Image")
question = gr.Textbox(label="Question", placeholder="Enter your question here")

# Output for the interface
answer = gr.Textbox(label="Predicted answer", show_label=True, show_copy_button=True)

# Title, description, and article for the interface
title = "Bio-Medical Question Answering"
description = "Gradio Demo for the Dragonfly-Med multimodal biomedical visual-language model, trained by instruction tuning on Llama 3.1. This model can answer bio-medical questions about images in natural language. To use it, upload your image, type a question about it, and click 'Submit', or click one of the examples to load them. You can read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2406.00977v2' target='_blank'>Dragonfly: Multi-Resolution Zoom-In Encoding Enhances Vision-Language Models</a> | <a href='https://huggingface.co/togethercomputer/Llama-3.1-8B-Dragonfly-Med-v2' target='_blank'>Model Page</a></p>"


# Launch the interface
interface = gr.Interface(
    fn=describe_image,
    inputs=[image, question],
    outputs=answer,
    cache_examples=True,
    title=title,
    description=description,
    article=article,
    theme="Base",
)
interface.launch(debug=False)
