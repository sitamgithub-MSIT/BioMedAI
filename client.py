import requests
from pprint import pprint

url = "http://localhost:8000/predict"

# Input image path for the test
image_path = "images/ROCO_04197.jpg"

# The actual question to ask the model
question = "Provide a brief description of the given image."

# Create the payload for the request
payload = {"image_path": image_path, "question": question}

# Send the request to the server and get the response
response = requests.post(url, json=payload)
pprint(response.json())
