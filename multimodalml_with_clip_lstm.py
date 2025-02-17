import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
#os.makedirs("/content/Image_Captioning/", exist_ok=True)
# %cd /content/Image_Captioning/
# os.makedirs("dataset", exist_ok=True)
# os.makedirs("checkpoints", exist_ok=True)
# !wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P dataset/
# !unzip -q dataset/Flickr8k_Dataset.zip -d dataset/
# !wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P dataset/
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
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self,sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenize(text)
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

        #print(self.imgs[:5])


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
    inp = inp.numpy().transpose((1,2,0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

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

'''
#generating the iterator from the dataloader
dataiter = iter(data_loader)

#getting the next batch
batch = next(dataiter)

#unpacking the batch
images, captions = batch

#showing info of image in single batch
for i in range(BATCH_SIZE):
    img,cap = images[i],captions[i]
#     print(f"captions - {captions[i]}")
    caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
    eos_index = caption_label.index('<EOS>')
    caption_label = caption_label[1:eos_index]
    caption_label = ' '.join(caption_label)
    show_image(img,caption_label)
    plt.show()
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import subprocess

# vgg16 = models.resnet50(pretrained=True)
# for param in vgg16.parameters():
#     param.requires_grad_(False)
# modules = list(vgg16.children())[:-1]
# print(dir(vgg16))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCLIP(nn.Module):
    def __init__(self,embed_size):
        super(EncoderCLIP, self).__init__()
        try:
            import clip
        except ModuleNotFoundError:
            # Install the CLIP module
            subprocess.check_call(["pip", "install", "git+https://github.com/openai/CLIP.git"])
            import clip
        self.model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model
        self.embed = nn.Linear(self.model.visual.output_dim, embed_size)

    def forward(self,images):
        with torch.no_grad():
            features = self.model.encode_image(images).float()  # Ensure features are in float type
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
        super(DecoderRNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,batch_first=True)
        self.fcn = nn.Linear(hidden_size,vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self,features, captions):
        # vectorize the caption
#         print(f"captions - {captions[:,:-1]}")
#         print(f"caption shape - {captions[:,:-1].shape}")
        embeds = self.embedding(captions[:,:-1])
#         print(f"shape of embeds - {embeds.shape}")
        # concat the features and captions
#         print(f"features shape - {features.shape}")
#         print(f"features unsqueeze at index 1 shape - {features.unsqueeze(1).shape}")
        x = torch.cat((features.unsqueeze(1),embeds),dim=1)
#         print(f"shape of x - {x.shape}")
        x,_ = self.lstm(x)
#         print(f"shape of x after lstm - {x.shape}")
        x = self.fcn(x)
#         print(f"shape of x after fcn - {x.shape}")
        return x

    def generate_caption(self,inputs,hidden=None,max_len=20,vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = inputs.size(0)

        captions = []

        for i in range(max_len):
            output,hidden = self.lstm(inputs,hidden)
            output = self.fcn(output)
            output = output.view(batch_size,-1)


            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated word
            captions.append(predicted_word_idx.item())

            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            #send generated word as the next caption
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))

        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions]
    
    def generate_caption2(self, inputs, hidden=None, max_len=20, vocab=None):
        """
        Generate a caption for an image using the trained model.
        
        Args:
            inputs (Tensor): The image feature vector of shape (1, embed_size).
            hidden (Tuple[Tensor, Tensor], optional): The initial hidden state for LSTM.
            max_len (int): Maximum caption length.
            vocab (Vocabulary): The vocabulary object.
        
        Returns:
            List[str]: Generated caption as a list of words.
        """
        captions = []

        # Ensure hidden state matches LSTM num_layers
        if hidden is None:
            hidden = (
                torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size).to(inputs.device),
                torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size).to(inputs.device)
            )

        for _ in range(max_len):
            output, hidden = self.lstm(inputs.unsqueeze(1), hidden)
            output = self.fcn(output.squeeze(1))  # Fully connected layer

            # Get the most probable word index
            predicted_word_idx = output.argmax(dim=1)

            # Save the generated word
            captions.append(predicted_word_idx.item())

            # Stop if <EOS> is predicted
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            # Use the predicted word as the next input
            inputs = self.embedding(predicted_word_idx)

        return [vocab.itos[idx] for idx in captions]


class EncoderDecoder(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
        super(EncoderDecoder,self).__init__()
        self.encoder = EncoderCLIP(embed_size)
        self.decoder = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers,drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
# resenet features shape - torch.Size([4, 2048, 1, 1])
# resenet features viewed shape - torch.Size([4, 2048])
# resenet features embed shape - torch.Size([4, 400])
# caption shape - torch.Size([4, 14])
# shape of embeds - torch.Size([4, 14, 400])
# features shape - torch.Size([4, 400])
# features unsqueeze at index 1 shape - torch.Size([4, 1, 400])
# shape of x - torch.Size([4, 15, 400])
# shape of x after lstm - torch.Size([4, 15, 512])
# shape of x after fcn - torch.Size([4, 15, 2994])
"""
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

# Hyperparameters
embed_size = 400
hidden_size = 512
vocab_size = len(dataset.vocab)
num_layers = 2
learning_rate = 0.0001
num_epochs = 1
print_every = 500

# initialize model, loss etc
model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Filter rows where the 'image' column contains a suffix after '.jpg'
invalid_images = df[df['image'].str.contains(r'\.jpg\.\d+$')]
print("Invalid image filenames:")
print(invalid_images['image'].tolist())


from google.colab import drive
drive.mount('/content/drive')
# Define the directory in Google Drive
CHECKPOINT_DIR = "/content/drive/My Drive/checkpoints/clip-lstm"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
from datetime import datetime

def save_checkpoint(epoch, model, optimizer, train_losses):
    '''Save model checkpoint.'''
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses  # Store list of batch losses for the epoch
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CHECKPOINT_DIR, f"{timestamp}_epoch_{epoch}.pth")
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# Track training losses across epochs
train_losses = []

for epoch in range(1,num_epochs+1):
    epoch_losses = []  # Store batch losses for this epoch
    for idx, (image, captions) in enumerate(iter(data_loader)):
        # if idx >= 2:
        #     print(f"stopped training at {idx} batches")
        #     break  # Stop after a few batches
        image,captions = image.to(device),captions.to(device)
        '''
        # Filter out placeholder data
        valid_indices = (images.sum(dim=[1, 2, 3]) != 0)
        images = images[valid_indices]
        captions = captions[valid_indices]

        if len(images) == 0:
            print(f"Batch {idx} skipped due to all missing images.")
            continue
        '''
        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs = model(image, captions)

        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))


        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Store loss for this batch
        epoch_losses.append(loss.item())

        if (idx+1)%print_every == 0:
            print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))

            #generate the caption
            model.eval()
            with torch.no_grad():
                dataiter = iter(data_loader)
                img,_ = next(dataiter)
                features = model.encoder(img[0:1].to(device))
                print(f"features shape - {features.shape}")
                caps = model.decoder.generate_caption(features.unsqueeze(0),vocab=dataset.vocab)
                caption = ' '.join(caps)
                print(caption)
                show_image(img[0],title=caption)

            model.train()

    # Append losses from this epoch to train_losses
    train_losses.append(epoch_losses)
    # Save checkpoint after each epoch
    save_checkpoint(epoch, model, optimizer, train_losses)

"""