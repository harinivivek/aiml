import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration  # Assuming you have a defined model class
from multimodalml_with_clip_transformer import EncoderDecoder, load_data

dataset = load_data()

if torch.cuda.is_available():
    device_name = "cuda"
#elif torch.backends.mps.is_available():
#    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

def load_model():
    embed_size = 400
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 2
    num_heads = 8
    dropout = 0.3
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout)
    checkpoint_path = "checkpoints/clip-transformer/20250216_115633_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    #model.load_state_dict(torch.load("/content/drive/My Drive/checkpoints/clip-transformer/20250216_115633_checkpoint.pth", map_location=device))
    #model.to(device)
    model.to(device).to(torch.float32)
    return model

def generate_caption_for_image(model, image, vocab):
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #image = Image.open(uploaded_file).convert("RGB")
    #image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    image_tensor = transform(image).unsqueeze(0).to(device).float()

    # Generate caption
    model.eval()
    with torch.no_grad():
        #features = model.encoder(image_tensor)
        features = model.encoder(image_tensor).to(torch.float32)
        print(f"Feature shape: {features.shape}")  # Debugging shape
        generated_tokens = model.decoder.generate_caption(features, vocab=vocab)
        generated_caption = ' '.join(generated_tokens)
    
    return generated_caption

# Streamlit UI
st.title("Image Caption Generator")
checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"

# Model selection dropdown
model_option = st.selectbox("Select a Model", ["Pretrained - BLIP Model", "Custom - CLIP Transformer Model"])

if model_option == "Pretrained - BLIP Model":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
elif model_option == "Custom - CLIP Transformer Model":
    model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if model_option == "Pretrained - BLIP Model":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    else:
        caption = generate_caption_for_image(model, image, dataset.vocab)
    st.write("### Generated Caption:")
    st.write(caption)
