import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.image as mpimg
import os
import shutil
import torch
from datetime import datetime

import os
from collections import Counter
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import torch.optim as optim
import os

# Assuming `dataset` is your image-caption dataset
from torch.utils.data import random_split, DataLoader

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

import matplotlib.pyplot as plt
spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freq_threshold):
        # Initialize the index-to-string (itos) dictionary with special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # Create the string-to-index (stoi) dictionary by reversing the itos dictionary
        self.stoi = {v: k for k, v in self.itos.items()}
        # Set the frequency threshold for adding words to the vocabulary
        self.freq_threshold = freq_threshold
        # text = "This is a good place to find a city"
        # [token.text.lower() for token in spacy_eng.tokenizer(text)]

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



# v = Vocabulary(freq_threshold=1)
# v.build_vocab(["This is a new city"])
# print(v.stoi)
# print(v.numericalize("This is a new city"))

# print(df["image"][0][::-1])

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

# vgg16 = models.resnet50(pretrained=True)
# for param in vgg16.parameters():
#     param.requires_grad_(False)
# modules = list(vgg16.children())[:-1]
# print(dir(vgg16))
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)
#device

class EncoderCLIP(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCLIP, self).__init__()
        try:
            import clip
        except ModuleNotFoundError:
            # Install the CLIP module
            !pip -q install git+https://github.com/openai/CLIP.git
            import clip  # Import CLIP
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

def save_checkpoint(epoch, model, optimizer, loss, train_losses, val_losses, bleu_scores, filename="checkpoint.pth", drive_enabled=False):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "bleu_scores": bleu_scores
    }
    
    if drive_enabled:
        save_dir = "/content/drive/My Drive/checkpoints/clip-transformer/"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{timestamp}_{filename}")
    else:
        save_path = filename
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch} to {save_path}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth", drive_enabled=False):
    if drive_enabled:
        from google.colab import drive
        #fixme: mount only if not already mounted
        drive.mount('/content/drive')
        !ls -ld "/content/drive/My Drive/"
        load_dir = "/content/drive/My Drive/checkpoints/clip-transformer/"
        if not os.path.isdir(load_dir):  # Check if directory exists
            print("No checkpoint dirs found. Starting from epoch 1.")
            return 1
        checkpoints = sorted(os.listdir(load_dir), reverse=True)
        if not checkpoints:
            print("No checkpoints found. Starting from epoch 1.")
            return 1
        load_path = os.path.join(load_dir, checkpoints[0])
    else:
        load_dir = "checkpoints"
        load_path = os.path.join(load_dir, "checkpoint.pth")
    
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        model.to(device)
        print("Loaded model state keys:", checkpoint["model_state"].keys())
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        bleu_scores = checkpoint.get("bleu_scores", [])
        print(f"Starting from epoch {start_epoch} as checkpoint file {load_path} loaded with epoch {start_epoch - 1}, with val_losses: {val_losses}")
        return start_epoch, train_losses, val_losses, bleu_scores
    
    print("Starting from epoch 1 as no saved checkpoint exists")
    return 1  # Start from epoch 1 if no checkpoint exists

#!pip uninstall torch torchtext -y
#torchtest dev has paused a year back so use nltk for bleu score
#!pip install torch torchtext --no-cache-dir
#!pip install torchtext --no-cache-dir



# **BLEU Score Calculation Function**
def calculate_bleu(reference, candidate):
    """Computes BLEU-4 score for generated captions."""
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    #from torchtext.data.metrics import bleu_score
    smoothie = SmoothingFunction().method4  # Smoothing function for better BLEU score
    return sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    #return bleu_score([reference], candidate)

def load_data():
    #os.makedirs("/content/Image_Captioning/", exist_ok=True)
    #%cd /content/Image_Captioning/

    os.makedirs("checkpoints", exist_ok=True)
        
    if os.path.exists('dataset'):
         #shutil.rmtree('dataset', ignore_errors=True)
        print("dataset folder already exists")
    else:
        os.makedirs("dataset", exist_ok=True)
        !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P dataset/
        !unzip -q dataset/Flickr8k_Dataset.zip -d dataset/
        !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P dataset/
        !unzip -q dataset/Flickr8k_text.zip -d dataset/
    
    
    image_data_location = os.path.join("dataset/Flicker8k_Dataset")
    # Ensure the directory exists before attempting to list its contents
    if os.path.exists(image_data_location):
        # Get a list of files in the directory
        files = [f for f in os.listdir(image_data_location) if os.path.isfile(os.path.join(image_data_location, f))]
    
        # Print the number of files
        print(f"Number of files in the directory: {len(files)}")
    else:
        print(f"Directory {image_data_location} does not exist.")
    caption_data_location = os.path.join("dataset/Flickr8k.token.txt")
    
    #image_data_location
    
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
    #cleaned_df['image'] = cleaned_df['image'].str.replace(r'(\.jpg).*$', r'\1', regex=True)
    #replaced above line with below to fix warning:
    #SettingWithCopyWarning: 
    # A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead
    # See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    cleaned_df.loc[:, 'image'] = cleaned_df['image'].str.replace(r'(\.jpg).*$', r'\1', regex=True)
    # Filter out rows where the image column has the specified value
    df = cleaned_df[cleaned_df['image'] != "2258277193_586949ec62.jpg"]
    # Reset index for a clean DataFrame
    df.reset_index(drop=True, inplace=True)
    #df.shape
    #df.head()
    # data_idx = 11
    # image_path = image_data_location + "/" + df.iloc[data_idx,0]
    # # print( df.iloc[data_idx,:])
    # img = mpimg.imread(image_path)
    # plt.imshow(img)
    # plt.show()
    
    # for i in range(data_idx, data_idx+5):
    #     print(f"Caption - {df.iloc[i,1]}")

    
    #defing the transform to be applied
    transforms = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])
    # testing the dataset
    dataset = CustomDataset(
            root_dir = image_data_location,
            df = df,
            transform = transforms
    )
    
    #img, caps = dataset[0]
    # print(caps)
    # show_image(img,"Image")
    # print("Token :",caps)
    # print("Sentence: ")
    # print([dataset.vocab.itos[token] for token in caps.tolist()])
    return dataset


def test_model_before_and_after_loading_checkpoint(model, log_prefix):
    print(f"weights {log_prefix}:")
    for name, param in model.named_parameters():
        if 'fc_out.weight' in name:  # Choose a key to inspect
            print(name, param.data.flatten()[:5])  # Print first 5 values
            break  # Only checking one layer for now
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)  # Dummy input
        features = model.encoder(test_input)  # Get encoded features
        #print("encoder features after method call: 'model.encoder(test_input)':", features)
        caps = model.decoder.generate_caption(features, vocab=dataset.vocab)  # Convert features to caption tokens
        print(f"{log_prefix}: Generated caption: {' '.join(caps)}")  # Convert indices to words

if __name__ == "__main__" or "google.colab" in str(get_ipython()):

    run_mode = input("Enter the run mode tt(for train_and_test)/t(for test only): ")
    early_stopping_enabled = input("enable early stopping(y/n): ").strip().lower() == "y"
    checkpoint_enabled = input("enable check points save/load(y/n): ").strip().lower() == "y"
    drive_enabled = input("enable google drive(y/n): ").strip().lower() == "y"
    num_epochs = int(input("no. of epochs: "))
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] started execution")

    dataset = load_data()
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



    # Hyperparameters
    embed_size = 400
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 2
    num_heads = 8
    dropout = 0.3
    learning_rate = 0.0001
    #num_epochs = 2

    # initialize model, loss etc
    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, num_heads, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Ensure output directory exists
    output_dir = "bleu_scores"
    os.makedirs(output_dir, exist_ok=True)

    # Store BLEU scores and losses
    train_losses = []
    val_losses = []
    bleu_scores = []
    start_epoch=1

    test_model_before_and_after_loading_checkpoint(model, "before loading checkpoint")
    # Load checkpoint if available
    if checkpoint_enabled:
        start_epoch, train_losses, val_losses, bleu_scores = load_checkpoint(model, optimizer, drive_enabled=drive_enabled)
    test_model_before_and_after_loading_checkpoint(model, "after loading checkpoint")

    #num_epochs = 2
    print_every = 500
    early_stopping = early_val_stopping = early_test_stopping = early_stopping_enabled
    max_batches = 2  # Limit the number of batches
    max_val_batches = 2  # Limit validation to a few batches to debug bleu score
    max_test_batches = 2  # Limit validation to a few batches to debug bleu score

    if run_mode == "tt":
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] started training with {num_epochs} epochs")
        # Training Loop
        for epoch in range(start_epoch, num_epochs + 1):
            model.train()
            running_loss = 0.0

            for idx, (image, captions) in enumerate(train_loader):
                if early_stopping and idx >= max_batches:
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
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # **Validation Step**
            model.eval()
            val_loss = 0.0
            total_bleu = 0
            with torch.no_grad():
                for idx, (image, captions) in enumerate(val_loader):
                    if early_val_stopping and idx >= max_val_batches:
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
                    if early_test_stopping:
                        show_image(img[0], title=caption)
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
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Final bleu: {bleu_score:.6%} Validation Avg BLEU-4: {avg_bleu:.6%}")

        print(f"Length of train_losses: {len(train_losses)}")
        print(f"Length of val_losses: {len(val_losses)}")
        print(f"Expected num_epochs: {num_epochs}")
        #Save checkpoint after each epoch
        #if checkpoint_enabled:
        save_checkpoint(epoch, model, optimizer, running_loss, train_losses, val_losses, bleu_scores, drive_enabled=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved after epoch {epoch}")


        try:
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
        except Exception as e:
            print(f"plot failed with exception: {e}")



    # **Testing Phase**
    model.eval()
    test_loss = 0.0
    total_test_bleu = 0
    with torch.no_grad():
        for idx, (image, captions) in enumerate(test_loader):
            if early_test_stopping and idx >= max_test_batches:
                print(f"stopped test at {idx} batches")
                break  # Stop early
            elif early_test_stopping:
                print(f"Test batch {idx}, batch size: {len(image)}")
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
                print(f"Generated Caption: {caption}")
                reference = [dataset.vocab.itos[idx] for idx in captions[i].cpu().numpy() if idx != 0]  # Ignore padding
                candidate = caption
                bleu_score = calculate_bleu(reference, candidate)
                total_test_bleu += bleu_score
            if early_test_stopping:
                show_image(img[0], title=caption)
        show_image(img[0], title=caption)
    num_test_samples = len(test_loader.dataset)
    avg_test_bleu = total_test_bleu / float(num_test_samples) if num_test_samples > 0 else 0  # Ensure float division to avoid int division resulting in zero always  
    print(f"Final Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final Test bleu score: {bleu_score:.6%}, Num Samples: {num_test_samples}, Test Avg BLEU-4 Score: {avg_test_bleu:.6%}")
