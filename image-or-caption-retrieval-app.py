import os
import torch
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import glob

# Set up directories
base_dir = os.path.dirname(os.path.abspath(__file__))
captions_embeddings_dir = f"{base_dir}/Flickr30k_Captions_Embeddings"
captions_file = f"{base_dir}/Flickr30k_Dataset/captions.txt"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# Function to get image embedding
def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding / torch.norm(image_embedding)

# Load captions from file
def load_captions(captions_file):
    captions = {}
    with open(captions_file, "r") as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            image_name, num, caption = line.strip().split(",", 2)
            base_name = image_name.split(".")[0]
            if base_name not in captions:
                captions[base_name] = []
            captions[base_name].append(caption)
    return captions

captions = load_captions(captions_file)
captions_filenames_list = glob.glob(os.path.join(captions_embeddings_dir, "*.pt"))

# Function to load embeddings lazily
def get_embedding_lazy(directory, filename):
    return torch.load(os.path.join(directory, filename))

# Search top matches
def search_top_matches_lazy(query_embedding, filenames_list, top_n):
    scores = []
    for filename in filenames_list:
        embedding = get_embedding_lazy(captions_embeddings_dir, filename).to(device)
        embedding = embedding / torch.norm(embedding)
        similarity = torch.matmul(embedding, query_embedding.T).squeeze().item()
        scores.append((os.path.basename(filename), similarity))

    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"item": match[0], "score": match[1]} for match in top_matches]

# Streamlit UI
st.title("Image Captioning with CLIP VIT")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Generating captions...")
    query_embedding = get_image_embedding(image)
    top_captions = search_top_matches_lazy(query_embedding, captions_filenames_list, top_n=5)

    st.write("### Top Matching Captions:")
    for i, item in enumerate(top_captions):
        caption_key = item['item'].replace('.pt', '')
        first_part, second_part = caption_key.rsplit('_', 1)
        caption_index = int(second_part) - 1
        caption_text = captions[first_part][caption_index]
        st.write(f"{i+1}. {caption_text} (Score: {item['score']:.4f})")
