import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import zipfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

import itertools
import os
import torch
from tqdm import tqdm
#from google.colab import auth
#from googleapiclient.discovery import build
import os
#from google.colab import auth
#from googleapiclient.discovery import build
#from googleapiclient.http import MediaIoBaseDownload
import io

# # Authenticate and initialize Google Drive API
# auth.authenticate_user()
# drive_service = build('drive', 'v3')

# # Step 1: Find MMRCS folder in "Shared with me"
# shared_files = drive_service.files().list(
#     q="sharedWithMe = true and mimeType = 'application/vnd.google-apps.folder'",
#     fields="files(id, name)").execute().get('files', [])

# mmrcs_folder = next((f for f in shared_files if f['name'] == 'MMRCS'), None)

# if not mmrcs_folder:
#     raise FileNotFoundError("MMRCS folder not found in 'Shared with me'.")

# mmrcs_folder_id = mmrcs_folder['id']
# print(f"MMRCS Folder ID: {mmrcs_folder_id}")

# # Step 2: Find both "Flickr30k_Images_Embeddings" and "Flickr30k_Captions_Embeddings" inside MMRCS
# mmrcs_contents = drive_service.files().list(
#     q=f"'{mmrcs_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'",
#     fields="files(id, name)").execute().get('files', [])

#!ls -ld "/content/drive/My Drive/"
# Fetch embeddings folders
#embeddings_folders = {f['name']: f"/content/drive//{f['name']}" for f in mmrcs_contents}
base_dir = os.path.dirname(os.path.abspath(__file__))
image_embeddings_dir = f"{base_dir}/Flickr30k_Images_Embeddings"
captions_embeddings_dir = f"{base_dir}/Flickr30k_Captions_Embeddings"

# Ensure both folders exist
# if 'Flickr30k_Images_Embeddings' not in embeddings_folders:
#     raise FileNotFoundError("'Flickr30k_Images_Embeddings' folder not found inside MMRCS.")
# if 'Flickr30k_Captions_Embeddings' not in embeddings_folders:
#     raise FileNotFoundError("'Flickr30k_Captions_Embeddings' folder not found inside MMRCS.")

# image_embeddings_dir = embeddings_folders['Flickr30k_Images_Embeddings']
# captions_embeddings_dir = embeddings_folders['Flickr30k_Captions_Embeddings']

print(f"Image Embeddings Directory: {image_embeddings_dir}")
print(f"Captions Embeddings Directory: {captions_embeddings_dir}")

# Ensure Drive is mounted
# if not os.path.ismount("/content/drive"):
#     from google.colab import drive
#     drive.mount("/content/drive")

# Step 3: Function to Load Embeddings
def load_embeddings(directory):
    filenames, embeddings = [], []
    for filename in tqdm(os.listdir(directory), desc=f"Processing {os.path.basename(directory)}"):
        if filename.endswith(".pt"):  # Assuming embeddings are stored as .pt files
            filepath = os.path.join(directory, filename)
            embedding = torch.load(filepath)
            filenames.append(filename)  # Remove file extension
            embeddings.append(embedding)
    embeddings_dict = {str(embed.tolist()): filename.split('.')[0] for embed, filename in zip(embeddings, filenames)}
    return filenames, embeddings, embeddings_dict

# Load image embeddings
image_filenames_list, image_embeddings_list, image_embeddings_dict = load_embeddings(image_embeddings_dir)

# Load captions embeddings
captions_filenames_list, captions_embeddings_list, captions_embeddings_dict = load_embeddings(captions_embeddings_dir)

def setup_logging(log_directory=f"{base_dir}/logs", log_file="mmrcs30k.log"):
    """
    Set up logging configuration with both console and file handlers.
    Args:
        log_directory (str): Directory where the log file will be saved.
        log_file (str): The name of the log file.
    Returns:
        logger: Configured logger object.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Full log file path
    log_path = os.path.join(log_directory, log_file)
    # Create a custom logger
    logger = logging.getLogger()
    # First, remove all existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Set the logging level to INFO
    logger.setLevel(logging.INFO)
    # Create handlers for file and console output
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    # Set the log level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
logger = setup_logging()

# Load the pre-trained CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(device)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to get embeddings for text using CLIP
def get_query_embedding(query_string):
    inputs = processor(text=query_string, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding / torch.norm(text_embedding)  # Normalize

# Function to get embeddings for Image using CLIP
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding / torch.norm(image_embedding)

def search_top_matches(query_input, embeddings_list, items_list, top_n, get_embedding_fn):
    """
    Generic function to search for top N matches based on cosine similarity.

    Parameters:
    - query_input (str or image_path): The input query (text caption or image path).
    - embeddings_list (list of Tensors): List of stored embeddings (image or caption).
    - items_list (list): List of items (image filenames or text captions).
    - top_n (int): Number of top matches to return.
    - get_embedding_fn (function): Function to get the embedding (e.g., get_image_embedding or get_query_embedding).

    Returns:
    - List of dictionaries with "item" and "score" keys.
    """
    # Get the query embedding and normalize it
    query_embedding = get_embedding_fn(query_input).to(device)
    query_embedding = query_embedding / torch.norm(query_embedding)

    # Convert list of embeddings into a tensor for batch processing
    embeddings_tensor = torch.stack(embeddings_list).to(device)

    # Normalize all stored embeddings
    embeddings_tensor = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)

    # Compute cosine similarity in batch
    similarities = torch.matmul(embeddings_tensor, query_embedding.T).squeeze()

    # Get indices of top N matches
    #top_indices = torch.argsort(similarities, descending=True)[:top_n]

    # Retrieve top matching items
    #top_matches = [{"item": items_list[i], "score": similarities[i].item()} for i in top_indices]

    # Get indices of top N+1 matches (to account for removing query itself)
    top_indices = torch.argsort(similarities, descending=True)

    # Retrieve top matching items, excluding the query image itself and scores >= 1
    top_matches = []
    query_index = items_list.index(query_input) if query_input in items_list else None  # Find query index if it exists

    for i in top_indices:
        if len(top_matches) >= top_n:
            break  # Stop once we have enough matches
        if query_index is not None and i == query_index:
            continue  # Skip the query image itself
        if similarities[i].item() >= 1:
            continue  # Skip matches with scores >= 1

        top_matches.append({"item": items_list[i], "score": similarities[i].item()})


    return top_matches

def display_image(image_path):
  img = Image.open(image_path)
  plt.figure(figsize=(2, 2))
  plt.imshow(img)
  plt.axis("off")  # Hide axes
  plt.show()

# dataset_folder = next((f for f in mmrcs_contents if f['name'] == 'Flickr30k_Dataset'), None)

# if not dataset_folder:
#     raise FileNotFoundError("'Flickr30k_Dataset' folder not found inside MMRCS.")

# dataset_folder_id = dataset_folder['id']

# # Step 3: Find "captions.txt" inside "Flickr30k_Dataset"
# dataset_contents = drive_service.files().list(
#     q=f"'{dataset_folder_id}' in parents and name = 'captions.txt'",
#     fields="files(id, name)").execute().get('files', [])

# captions_file_entry = dataset_contents[0] if dataset_contents else None

# if not captions_file_entry:
#     raise FileNotFoundError("'captions.txt' file not found inside Flickr30k_Dataset.")

# captions_file_id = captions_file_entry['id']
captions_file = f"{base_dir}/Flickr30k_Dataset/captions.txt"  # Local path to store the downloaded file

# # Step 4: Download "captions.txt" from Google Drive
# request = drive_service.files().get_media(fileId=captions_file_id)
# with open(captions_file, "wb") as f:
#     downloader = MediaIoBaseDownload(f, request)
#     done = False
#     while not done:
#         status, done = downloader.next_chunk()
#         print(f"Downloading captions.txt: {int(status.progress() * 100)}%")

# print(f"Captions file downloaded: {captions_file}")

# Load captions from the file
def load_captions(captions_file):
    captions = {}
    with open(captions_file, "r") as f:
      lines = f.readlines()
      for index, line in enumerate(lines):
        if index == 0:
          continue
        image_name, num, caption = line.strip().split(",",2 )
        # Remove the file extension at the end of the image name
        base_name = image_name.split(".")[0]
        if base_name not in captions:
           captions[base_name] = []
        captions[base_name].append(caption)
      return captions

#captions_file = "/content/Flickr30k_Dataset/captions.txt"  # Path to captions file
captions = load_captions(captions_file)

images_path = f"{base_dir}/Flickr30k_Dataset/flickr30k/images"  # Path to images

#Search for the top captions matching a given image.
def imageToCaptions(image_name):
  top_n = 5
  logger.info("Image---to---Captions")
  image_path = os.path.join(images_path, image_name)
  display_image(image_path)
  logger.info(f"Top {top_n} captions:")
  top_captions = search_top_matches(image_path,captions_embeddings_list,captions_filenames_list,top_n,get_image_embedding)
  # Print top 5 results
  for i, item in enumerate(top_captions):
    caption = item['item'].replace('.pt', '')  # Remove '.pt'
    first_part, second_part = caption.rsplit('_', 1)  # Split at last underscore
    second_part = int(second_part)-1;
    logger.info(f"{i+1}.{captions[first_part][second_part]}, Score: {item['score']:.4f}")
  logger.info(f"completed")

image_name="1026792563.jpg"
imageToCaptions(image_name)