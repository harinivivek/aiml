import streamlit as st
import torch
import os
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration  # Assuming you have a defined model class
from multimodalml_with_clip_transformer import EncoderDecoder as clipTransformer, load_data
from multimodal_resnet_lstm import EncoderDecoder as resnetLstm, dataset as resnetDataset



if torch.cuda.is_available():
    device_name = "cuda"
#elif torch.backends.mps.is_available():
#    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

def load_model(dataset):
    embed_size = 400
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 2
    num_heads = 8
    dropout = 0.3
    model = clipTransformer(embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout)
    checkpoint_path = "checkpoints/clip-transformer/20250216_115633_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    #model.load_state_dict(torch.load("/content/drive/My Drive/checkpoints/clip-transformer/20250216_115633_checkpoint.pth", map_location=device))
    #model.to(device)
    model.to(device).to(torch.float32)
    return model

def resnet_lstm_load_model(dataset):
    # Hyperparameters
    embed_size = 400
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 0.0001
    model = resnetLstm(embed_size, hidden_size, vocab_size, num_layers)
    checkpoint_path = "checkpoints/resnet-lstm/20250216_155414_epoch_1.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    #model.load_state_dict(torch.load("/content/drive/My Drive/checkpoints/clip-transformer/20250216_115633_checkpoint.pth", map_location=device))
    #model.to(device)
    model.to(device).to(torch.float32)
    return model

def generate_caption_for_image(model, image, vocab, transform, is_resnet):
    # Load and preprocess image
    #image = Image.open(uploaded_file).convert("RGB")
    #image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device).float()

    # Generate caption
    model.eval()
    with torch.no_grad():
        #features = model.encoder(image_tensor)
        features = model.encoder(image_tensor).to(torch.float32)
        print(f"Feature shape: {features.shape}")  # Debugging shape
        if is_resnet:
            # Initialize the LSTM hidden state (num_layers, batch_size, hidden_size)
            batch_size = features.size(0)  # Should be 1 for a single image
            hidden_size = model.decoder.lstm.hidden_size
            num_layers = model.decoder.lstm.num_layers

            # Ensure the hidden state has the correct shape
            hidden = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                    torch.zeros(num_layers, batch_size, hidden_size).to(device))

            #generated_tokens = model.decoder.generate_caption(features, hidden=hidden, vocab=vocab)
            generated_tokens = model.decoder.generate_caption2(features, vocab=vocab)
        else:
            generated_tokens = model.decoder.generate_caption(features, vocab=vocab)

        generated_caption = ' '.join(generated_tokens)
    
    return generated_caption

# Streamlit UI
st.title("Image Caption Generator")
checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
is_resnet = False

# Model selection dropdown
model_option = st.selectbox("Select a Model", ["Pretrained - BLIP Model", "CLIP + Transformer Model", "RESNET50 + LSTM Model"])

if model_option == "Pretrained - BLIP Model":
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    dataset = load_data()
elif model_option == "CLIP + Transformer Model":
    dataset = load_data()
    model = load_model(dataset)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
elif model_option == "RESNET50 + LSTM Model":
    dataset = resnetDataset
    model = resnet_lstm_load_model(dataset)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    is_resnet = True
    

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
        caption = generate_caption_for_image(model, image, dataset.vocab, transform, is_resnet)
    st.write("### Generated Caption:")
    st.write(caption)
