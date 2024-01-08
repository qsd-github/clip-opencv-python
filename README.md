# clip-opencv-python
Modified the official source code of OpenAI to enable clip not only to support PIL, but also to support opencv-python

# usage
```python
import clip
import torch
import cv2 as cv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("./ViT-B-32.pt", device=device)

image = preprocess(cv.imread("./images/dog.jpg")).unsqueeze(0).to(device) 
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print(probs)
```
