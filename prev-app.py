import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration  # Assuming you have a defined model class
from multimodalml_with_clip_transformer import EncoderDecoder, dataset

def load_model_from_checkpoint(checkpoint_dir):
    """Load the most recent model checkpoint."""
    checkpoints = sorted(os.listdir(checkpoint_dir), reverse=True)
    if not checkpoints:
        st.error("No checkpoints found!")
        return None
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Hyperparameters
    embed_size = 400
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 2
    num_heads = 8
    dropout = 0.3
    learning_rate = 0.0001
    num_epochs = 2

    # initialize model, loss etc
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout).to(device)

    #model = EncoderDecoder()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    st.success(f"Loaded model from {checkpoint_path}")
    return model

def predict_caption(model, image, vocab):
    """Generate a caption using the custom EncoderDecoder model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        features = model.encoder(image)
        print(features)
        caption = model.decoder.generate_caption(features.unsqueeze(0), vocab=vocab)
    return ' '.join(caption)

# Streamlit UI
st.title("Image Captioning with Pretrained Model")
checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"

# Model selection dropdown
model_option = st.selectbox("Select a Model", ["BLIP Image Captioning", "Custom ImageCaptioningModel"])

if model_option == "BLIP Image Captioning":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
elif model_option == "Custom ImageCaptioningModel":
    model = load_model_from_checkpoint(checkpoint_dir)
    processor = None  # Custom model doesn't need a processor
    vocab = dataset.vocab  # Assuming dataset object with vocabulary is available

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if model_option == "BLIP Image Captioning":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    else:
        caption = predict_caption(model, image, vocab)
    st.write("### Generated Caption:")
    st.write(caption)
