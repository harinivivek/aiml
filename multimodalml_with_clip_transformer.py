import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
import torch

# Install the CLIP module
#!pip install git+https://github.com/openai/CLIP.git

import clip  # Import CLIP

os.makedirs("/content/Image_Captioning/", exist_ok=True)
%cd /content/Image_Captioning/
# if os.path.exists('dataset'):
#     shutil.rmtree('dataset', ignore_errors=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# shutil.rmtree('dataset/__MACOSX', ignore_errors=True)
# if os.path.exists('dataset/Flickr8k_Dataset.zip'):
#     os.remove('dataset/Flickr8k_Dataset.zip')
# if os.path.exists('dataset/Flickr8k_text.zip'):
#     os.remove('dataset/Flickr8k_text.zip')

# !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P dataset/

# !unzip -q dataset/Flickr8k_Dataset.zip -d dataset/



# !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P dataset/

# !unzip -q dataset/Flickr8k_text.zip -d dataset/

# shutil.rmtree('dataset/__MACOSX', ignore_errors=True)
# if os.path.exists('dataset/Flickr8k_Dataset.zip'):
#     os.remove('dataset/Flickr8k_Dataset.zip')
# if os.path.exists('dataset/Flickr8k_text.zip'):
#     os.remove('dataset/Flickr8k_text.zip')

image_data_location = os.path.join("dataset/Flicker8k_Dataset")
# Get a list of files in the directory
files = [f for f in os.listdir(image_data_location) if os.path.isfile(os.path.join(image_data_location, f))]

# Print the number of files
print(f"Number of files in the directory: {len(files)}")
caption_data_location = os.path.join("dataset/Flickr8k.token.txt")

image_data_location

captions_data = []
with open(caption_data_location, 'r') as file:
    for line in file:
        # Split lines based on your format
        row = line.strip().split()  # Adjust split logic as needed
        captions_data.append(row)

# Convert to DataFrame
df1 = pd.DataFrame(captions_data)
df1.head()
print(df1.shape)

# Combine caption parts (columns 2 onwards) into a single string
df1["caption"] = df1.iloc[:, 1:].apply(lambda x: " ".join(filter(None, x)), axis=1)
# Extract image name by removing the '#<number>' suffix
df1["image"] = df1[0].str.split("#").str[0]
# Keep only the relevant columns
cleaned_df = df1[["image", "caption"]]
# Remove anything after .jpg in the 'image' column
cleaned_df['image'] = cleaned_df['image'].str.replace(r'(\.jpg).*$', r'\1', regex=True)
# Filter out rows where the image column has the specified value
df = cleaned_df[cleaned_df['image'] != "2258277193_586949ec62.jpg"]
# Reset index for a clean DataFrame
df.reset_index(drop=True, inplace=True)
df.shape



df.head()

data_idx = 11
image_path = image_data_location + "/" + df.iloc[data_idx,0]
# print( df.iloc[data_idx,:])
img = mpimg.imread(image_path)
plt.imshow(img)
plt.show()

for i in range(data_idx, data_idx+5):
    print(f"Caption - {df.iloc[i,1]}")

import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

spacy_eng = spacy.load('en_core_web_sm')
text = "This is a good place to find a city"
[token.text.lower() for token in spacy_eng.tokenizer(text)]

class Vocabulary:
    def __init__(self, freq_threshold):
        # Initialize the index-to-string (itos) dictionary with special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # Create the string-to-index (stoi) dictionary by reversing the itos dictionary
        self.stoi = {v: k for k, v in self.itos.items()}
        # Set the frequency threshold for adding words to the vocabulary
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    #The frequency threshold is used to filter out infrequent words from the vocabulary. 
    # Words that appear less frequently than the threshold are not added to the vocabulary.
    #  This helps in reducing the size of the vocabulary and focusing on more common words, 
    # which can improve the efficiency and performance of the model. 
    # When a word's frequency reaches the threshold, it is added to the stoi and itos dictionaries 
    # to ensure that only sufficiently frequent words are included in the vocabulary.
    def build_vocab(self,sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    # Add the word to the stoi dictionary with a unique index
                    self.stoi[word] = idx
                    # Add the index-to-word mapping to the itos dictionary
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenize(text)
        # Convert each token to its corresponding index using the stoi dictionary
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]



v = Vocabulary(freq_threshold=1)
v.build_vocab(["This is a new city"])
print(v.stoi)
print(v.numericalize("This is a new city"))

print(df["image"][0][::-1])

class CustomDataset(Dataset):
    def __init__(self,root_dir,df=None,transform=None, freq_threshold=5):
        self.root_dir = root_dir
        #self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.df = df
        if self.df is None:
            raise ValueError("A valid DataFrame must be provided!")


        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        print(self.imgs[:5])


        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]

        img_location = os.path.join(self.root_dir,img_name)
        #print("img_location", img_location)
        try:
          img = Image.open(img_location).convert("RGB")
          if self.transform is not None:
            img = self.transform(img)
        except FileNotFoundError:
          print(f"File {img_location} not found. Returning placeholder image.")
          # Return a placeholder image tensor
          img =  torch.zeros(3, 224, 224)

        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        # If the image is a placeholder, return a placeholder caption tensor
        if img.sum() == 0:
            return img, torch.zeros(len(caption_vec), dtype=torch.long)
        return img, torch.tensor(caption_vec,dtype=torch.long)

#defing the transform to be applied
transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

def show_image(inp, title=None):
    """Imshow for Tensor"""
     # Check if input is on the GPU, and move it to CPU if true
    if inp.is_cuda:
        inp = inp.cpu()

    # If it's a batch of images, squeeze the first image (assuming batch size = 1)
    if len(inp.shape) == 4:
        inp = inp.squeeze(0)  # Removing batch dimension

    # Ensure the tensor is in the shape (C, H, W)
    if len(inp.shape) == 3 and inp.shape[0] == 3:  # Color image (3 channels)
        image = inp.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    elif len(inp.shape) == 3 and inp.shape[0] == 1:  # Grayscale image (1 channel)
        image = inp.numpy().squeeze(0)  # Convert from (1, H, W) to (H, W)
    elif len(inp.shape) == 2:  # If the image is 2D (grayscale without channel dimension)
        image = inp.numpy()  # No need to transpose, it's already (H, W)
    else:
        raise ValueError("Input tensor must have shape (C, H, W), where C is 1 or 3.")

    # Display the image
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)  # Use grayscale colormap if it's a single channel
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

# testing the dataset
dataset = CustomDataset(
        root_dir = image_data_location,
        df = df,
        transform = transforms
)

img, caps = dataset[0]
# print(caps)
show_image(img,"Image")
print("Token :",caps)
print("Sentence: ")
print([dataset.vocab.itos[token] for token in caps.tolist()])

class CapsCollate:
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        #print(f"shape - {(imgs)}")
        #print("----"*22)
        imgs = torch.cat(imgs,dim=0)
        #print(f"shape - {imgs}")
        #print("------")
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

#writing the dataloader
#setting the constants
BATCH_SIZE = 4
NUM_WORKER = 1

#token to represent the padding
pad_idx = dataset.vocab.stoi["<PAD>"]

data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)

#generating the iterator from the dataloader
dataiter = iter(data_loader)

#getting the next batch
batch = next(dataiter)

#unpacking the batch
images, captions = batch

#showing info of image in single batch
# for i in range(BATCH_SIZE):
#     img,cap = images[i],captions[i]
#     print(f"captions - {captions[i]}")
#     print(f"image - {images[i]}")
#     caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
#     eos_index = caption_label.index('<EOS>')
#     caption_label = caption_label[1:eos_index]
#     caption_label = ' '.join(caption_label)
#     show_image(img,caption_label)
#     plt.show()



import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# vgg16 = models.resnet50(pretrained=True)
# for param in vgg16.parameters():
#     param.requires_grad_(False)
# modules = list(vgg16.children())[:-1]
# print(dir(vgg16))

class EncoderCLIP(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCLIP, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model
        self.embed = nn.Linear(self.model.visual.output_dim, embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
        features = self.embed(features)
        return features

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_size))
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True  # Ensure batch_first is set to True
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        captions_embed = self.embedding(captions) + self.positional_encoding[:, :captions.size(1), :]
        features = features.unsqueeze(1).expand(-1, captions.size(1), -1)  # Ensure features have the same batch size as captions
        transformer_out = self.transformer(features, captions_embed)
        outputs = self.fc_out(transformer_out)
        return outputs

    def generate_caption(self, features, max_len=20, vocab=None):
        features = features.unsqueeze(0)  # Ensure features have the correct dimensions
        captions = [vocab.stoi["<SOS>"]]
        for _ in range(max_len):
            captions_tensor = torch.tensor(captions).unsqueeze(0).to(features.device)
            captions_embed = self.embedding(captions_tensor) + self.positional_encoding[:, :captions_tensor.size(1), :]
            transformer_out = self.transformer(features, captions_embed)
            output = self.fc_out(transformer_out[:, -1, :])
            predicted_word_idx = output.argmax(dim=1).item()
            captions.append(predicted_word_idx)
            if vocab.itos[predicted_word_idx] == "<EOS>":
                break
        return [vocab.itos[idx] for idx in captions]

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCLIP(embed_size)
        self.decoder = DecoderTransformer(embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

import torch
import torch.optim as optim
import os

# Assuming `dataset` is your image-caption dataset
from torch.utils.data import random_split, DataLoader

# Define dataset sizes
train_size = int(0.8 * len(dataset))  # 80% training
val_size = int(0.1 * len(dataset))    # 10% validation
test_size = len(dataset) - train_size - val_size  # 10% test

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Checkpoint function
def save_checkpoint(epoch, model, optimizer, loss, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded from epoch {start_epoch - 1}")
        return start_epoch
    return 1  # Start from epoch 1 if no checkpoint exists

#!pip uninstall torch torchtext -y
#torchtest dev has paused a year back so use nltk for bleu score
#!pip install torch torchtext --no-cache-dir
#!pip install torchtext --no-cache-dir


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#from torchtext.data.metrics import bleu_score

# Ensure output directory exists
output_dir = "bleu_scores"
os.makedirs(output_dir, exist_ok=True)

# **BLEU Score Calculation Function**
def calculate_bleu(reference, candidate):
    """Computes BLEU-4 score for generated captions."""
    smoothie = SmoothingFunction().method4  # Smoothing function for better BLEU score
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    #return bleu_score([reference], candidate)


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
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import matplotlib.pyplot as plt

# Store BLEU scores and losses
train_losses = []
val_losses = []
bleu_scores = []

# Load checkpoint if available
#start_epoch = load_checkpoint()
start_epoch=1

num_epochs = 1
print_every = 1
max_batches = 2  # Limit the number of batches
max_val_batches = 2  # Limit validation to a few batches to debug bleu score
max_test_batches = 2  # Limit validation to a few batches to debug bleu score

print(f"started training with {num_epochs} epochs")
# Training Loop
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for idx, (image, captions) in enumerate(train_loader):
        if idx >= max_batches:
            print(f"stopped training at {idx} batches")
            break  # Stop after a few batches
        image, captions = image.to(device), captions.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(image, captions)

        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))  # Adjust target captions
        loss.backward()

        # Update weights
        optimizer.step()
        running_loss += loss.item()

        if (idx + 1) % print_every == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    # Save checkpoint after each epoch
    #save_checkpoint(epoch, model, optimizer, running_loss)

    # **Validation Step**
    model.eval()
    val_loss = 0.0
    total_bleu = 0
    with torch.no_grad():
        for idx, (image, captions) in enumerate(val_loader):
            if idx >= max_val_batches:
               print(f"stopped val at {idx} batches")
               break  # Stop early
            image, captions = image.to(device), captions.to(device)
            outputs = model(image, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))  # Adjust target captions
            val_loss += loss.item()
            for img, cap in zip(image, captions):
                features = model.encoder(img.unsqueeze(0))#img[0:1].to(device))
                caps = model.decoder.generate_caption(features, vocab=dataset.vocab)  # Adjust features dimensions
                caption = ' '.join(caps)
                #print(f"Generated Caption: {caption}")
                #print(f"Shape of img[0]: {img[0].shape}")
                reference = [dataset.vocab.itos[idx] for idx in cap.cpu().numpy() if idx != 0]  # Ignore padding
                candidate = caption
                bleu_score = calculate_bleu(reference, candidate)
                total_bleu += bleu_score
            show_image(img[0], title=caption)

    num_val_samples = len(val_loader.dataset)
    #print(f"Avg Validation Loss after Epoch [{epoch}/{num_epochs}]: {val_loss / len(val_loader):.4f}")

    # **Generate a caption for an image from the validation set**
    #with torch.no_grad():
     #   val_iter = iter(val_loader)
      #  img, _ = next(val_iter)


    avg_val_loss = val_loss / len(val_loader)
    #avg_bleu = total_bleu / len(val_loader.dataset)
    avg_bleu = total_bleu / float(num_val_samples) if num_val_samples > 0 else 0  # Ensure float division to avoid int division resulting in zero always  
    print(f"Total BLEU: {total_bleu}, Num Samples: {num_val_samples}, avg_bleu: {avg_bleu} ")
    val_losses.append(avg_val_loss)
    bleu_scores.append(avg_bleu)
    #print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Final bleu: {bleu_score} Validation Avg BLEU-4: {avg_bleu:.4f}")
    #avg bleu score is very small so show 6 decimals and as percent
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Final bleu: {bleu_score:.6%} Validation Avg BLEU-4: {avg_bleu:.6%}")

print(f"Length of train_losses: {len(train_losses)}")
print(f"Length of val_losses: {len(val_losses)}")
print(f"Expected num_epochs: {num_epochs}")


# **Plot BLEU-4 Score and Loss over Epochs**
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Loss (left y-axis)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='red')
ax1.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-', color='red', label='Train Loss')
ax1.plot(range(1, num_epochs+1), val_losses, marker='s', linestyle='--', color='orange', label='Val Loss')
ax1.tick_params(axis='y', labelcolor='red')

# Plot BLEU-4 Score (right y-axis)
ax2 = ax1.twinx()
ax2.set_ylabel('BLEU-4 Score', color='blue')
ax2.plot(range(1, num_epochs+1), bleu_scores, marker='D', linestyle='-', color='blue', label='BLEU-4 Score')
ax2.tick_params(axis='y', labelcolor='blue')

# Legends and Title
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Loss and BLEU-4 Score Progression')
plt.grid()
plt.show()



# **Testing Phase**
model.eval()
test_loss = 0.0
with torch.no_grad():
    for idx, (image, captions) in enumerate(test_loader):
        if idx >= max_test_batches:
            print(f"stopped test at {idx} batches")
            break  # Stop early
        image, captions = image.to(device), captions.to(device)
        outputs = model(image, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))  # Adjust target captions
        test_loss += loss.item()
        ## Compute BLEU score
        for i in range(len(image)):
            img = image[i:i+1]
            features = model.encoder(img)
            caps = model.decoder.generate_caption(features, vocab=dataset.vocab)  # Adjust features dimensions
            caption = ' '.join(caps)
            #print(f"Generated Caption: {caption}")
            reference = [dataset.vocab.itos[idx] for idx in captions[i].cpu().numpy() if idx != 0]  # Ignore padding
            candidate = caption
            bleu_score = calculate_bleu(reference, candidate)
            total_bleu += bleu_score
        show_image(img[0], title=caption)
avg_test_bleu = total_bleu / len(test_loader.dataset)
print(f"Final Test Loss: {test_loss / len(test_loader):.4f}")
print(f"Final Test bleu score: {bleu_score} Avg BLEU-4 Score: {avg_test_bleu:.4f}")
