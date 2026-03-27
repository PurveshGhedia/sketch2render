import base64
import requests
from PIL import Image
import io
import numpy as np

# load pair and crop only the LEFT half (sketch)
pair = Image.open("data/test_pairs/00004.jpg")
w, h = pair.size
sketch = pair.crop((0, 0, w // 2, h))  # left half = sketch
sketch = sketch.resize((256, 256))

# encode to base64
buffer = io.BytesIO()
sketch.save(buffer, format="PNG")
encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

# send to API
response = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"image": encoded}
)

print("Status code:", response.status_code)

result = response.json()
img_data = base64.b64decode(result["image"])
img = Image.open(io.BytesIO(img_data))
img.save("api/test_output.png")
print("Saved output to api/test_output.png")
