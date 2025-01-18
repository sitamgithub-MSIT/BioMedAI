# Import necessary libraries
import sys
import os

# Add the Dragonfly directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Dragonfly'))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from src.dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
from src.dragonfly.models.processing_dragonfly import DragonflyProcessor
import litserve as ls


class DragonflyMedAPI(ls.LitAPI):
    """
    DragonflyMedAPI is a subclass of ls.LitAPI that provides an interface to the Dragonfly-Med family of models.

    Methods:
        - setup(device): Initializes the model and processor with the specified device.
        - decode_request(request): Decodes the incoming request to extract the inputs.
        - predict(data): Uses the model to generate a response for the given input.
        - encode_response(output): Encodes the generated response into a JSON format.
    """

    def setup(self, device):
        """
        Sets up the model and processor for the task.
        """
        # Set up model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "togethercomputer/Llama-3.1-8B-Dragonfly-Med-v2"
        )
        clip_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )
        image_processor = clip_processor.image_processor
        self.processor = DragonflyProcessor(
            image_processor=image_processor,
            tokenizer=self.tokenizer,
            image_encoding_style="llava-hd",
        )

        # Load the model
        self.model = DragonflyForCausalLM.from_pretrained(
            "togethercomputer/Llama-3.1-8B-Dragonfly-Med-v2"
        )
        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(device)

    def decode_request(self, request):
        """
        Decodes the input request to extract the image path and text prompt.
        """
        # Decode the request data
        image_path = request.get("image_path", None)
        question = request.get("question", None)
        text_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return image_path, text_prompt

    def predict(self, data):
        """
        Generates a response for the given input.
        """
        # Load and process image
        image_path, text_prompt = data
        image = Image.open(image_path)
        image = image.convert("RGB")
        images = [image]

        # Process input for the model
        inputs = self.processor(
            text=[text_prompt],
            images=images,
            max_length=1024,
            return_tensors="pt",
            is_generate=True,
        )
        inputs = inputs.to(self.device)

        # Generate output with the model
        with torch.inference_mode():
            return self.model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=self.tokenizer.encode("<|eot_id|>"),
                do_sample=False,
                use_cache=True,
            )

    def encode_response(self, output):
        """
        Encodes the given results into a dictionary format.
        """
        # Encode the response as JSON
        generation_text = self.processor.batch_decode(output, skip_special_tokens=True)
        assistant_response = generation_text[0].split("assistant\n\n", 1)[1].strip()
        return {"response": assistant_response}


if __name__ == "__main__":
    # Create an instance of the DragonflyMedAPI and run the LitServer
    api = DragonflyMedAPI()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
