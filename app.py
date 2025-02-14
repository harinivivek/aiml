import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration  # Assuming you have a defined model class

# def load_model_from_checkpoint(checkpoint_dir):
#     """Load the most recent model checkpoint."""
#     checkpoints = sorted(os.listdir(checkpoint_dir), reverse=True)
#     if not checkpoints:
#         st.error("No checkpoints found!")
#         return None
#     checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     model = ImageCaptioningModel()
#     model.load_state_dict(checkpoint['model_state'])
#     model.eval()
#     st.success(f"Loaded model from {checkpoint_path}")
#     return model

# def predict_caption(model, image):
#     """Generate a caption for the uploaded image."""
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         caption = model.generate_caption(image)  # Assuming your model has a caption generation method
#     return caption

# # Streamlit UI
# st.title("Image Captioning with Pretrained Model")
# checkpoint_dir = "/content/drive/My Drive/checkpoint/clip-transformer/"
# model = load_model_from_checkpoint(checkpoint_dir)

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
# if uploaded_file and model:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     caption = predict_caption(model, image)
#     st.write("### Generated Caption:")
#     st.write(caption)

import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration  # Assuming you have a defined model class

def load_model_from_checkpoint(checkpoint_dir):
    """Load the most recent model checkpoint."""
    checkpoints = sorted(os.listdir(checkpoint_dir), reverse=True)
    if not checkpoints:
        st.error("No checkpoints found!")
        return None
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = ImageCaptioningModel()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    st.success(f"Loaded model from {checkpoint_path}")
    return model

def predict_caption(model, processor, image):
    """Generate a caption for the uploaded image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption

# Streamlit UI
st.title("Image Captioning with Pretrained Model")
checkpoint_dir = "/content/drive/My Drive/checkpoint/clip-transformer/"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    caption = predict_caption(model, processor, image)
    st.write("### Generated Caption:")
    st.write(caption)
