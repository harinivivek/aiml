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
import os
import io
import glob


base_dir = os.path.dirname(os.path.abspath(__file__))
image_embeddings_dir = f"{base_dir}/Flickr30k_Images_Embeddings"
captions_embeddings_dir = f"{base_dir}/Flickr30k_Captions_Embeddings"

print(f"Image Embeddings Directory: {image_embeddings_dir}")
print(f"Captions Embeddings Directory: {captions_embeddings_dir}")

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
#image_filenames_list, image_embeddings_list, image_embeddings_dict = load_embeddings(image_embeddings_dir)

# Load captions embeddings
#captions_filenames_list, captions_embeddings_list, captions_embeddings_dict = load_embeddings(captions_embeddings_dir)

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
device = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(device)
if "model" not in globals():
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
    logger.info("Computing query embedding...")
    query_embedding = get_embedding_fn(query_input).to(device)
    query_embedding = query_embedding / torch.norm(query_embedding)

    # Convert list of embeddings into a tensor for batch processing
    logger.info("Stacking embeddings tensor...")
    embeddings_tensor = torch.stack(embeddings_list).to(device)

    # Normalize all stored embeddings
    logger.info("Normalizing stored embeddings...")
    embeddings_tensor = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)

    # Compute cosine similarity in batch
    logger.info("Computing cosine similarity...")
    similarities = torch.matmul(embeddings_tensor, query_embedding.T).squeeze()

    logger.info("Sorting similarities...")
    top_indices = torch.argsort(similarities, descending=True)

    # Retrieve top matching items, excluding the query image itself and scores >= 1
    logger.info("Retrieving top matches...")
    top_matches = []
    query_index = items_list.index(query_input) if query_input in items_list else None  # Find query index if it exists

    max_attempts = 100  # Avoid infinite loops
    attempts = 0
    for i in top_indices:
        if len(top_matches) >= top_n:
            break  # Stop once we have enough matches
        if attempts >= max_attempts:
            logger.warning("Max attempts reached. Breaking loop.")
            break
        if query_index is not None and i == query_index:
            continue  # Skip the query image itself
        if similarities[i].item() >= 1:
            continue  # Skip matches with scores >= 1

        top_matches.append({"item": items_list[i], "score": similarities[i].item()})


    return top_matches

def load_embeddings_lazily(directory):
    #filenames = os.listdir(directory)
    filenames = glob.glob(os.path.join(directory, "*.pt")) # Faster than os.listdir()
    return filenames  # Return just filenames, not actual tensors

#image_filenames_list = load_embeddings_lazily(image_embeddings_dir)
captions_filenames_list = load_embeddings_lazily(captions_embeddings_dir)

def get_embedding_lazy(directory, filename):
    return torch.load(os.path.join(directory, filename))

def search_top_matches_lazy(query_input, filenames_list, top_n, get_embedding_fn):
    logger.info("Computing query embedding...")
    query_embedding = get_embedding_fn(query_input).to(device)
    query_embedding = query_embedding / torch.norm(query_embedding)

    scores = []
    for filename in tqdm(filenames_list, desc="Computing similarities"):
        embedding = get_embedding_lazy(image_embeddings_dir, filename).to(device)
        embedding = embedding / torch.norm(embedding)
        similarity = torch.matmul(embedding, query_embedding.T).squeeze().item()
        scores.append((os.path.basename(filename), similarity))

    logger.info("Sorting similarities...")
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"item": match[0], "score": match[1]} for match in top_matches]

def display_image(image_path):
  img = Image.open(image_path)
  plt.figure(figsize=(2, 2))
  plt.imshow(img)
  plt.axis("off")  # Hide axes
  plt.show()

captions_file = f"{base_dir}/Flickr30k_Dataset/captions.txt"  # Local path to store the downloaded file


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
def imageToCaptions(image_path):
  top_n = 5
  logger.info("Image---to---Captions for image path: {image_path}")
  #display_image(image_path)
  logger.info(f"Top {top_n} captions:")
  #top_captions = search_top_matches(image_path,captions_embeddings_list,captions_filenames_list,top_n,get_image_embedding)
  top_captions = search_top_matches_lazy(image_path, captions_filenames_list, top_n, get_image_embedding)
  # Print top 5 results
  #logger.info(f"top_captions: {top_captions}")
  for i, item in enumerate(top_captions):
    #logger.info(f"item: {item}")
    caption = item['item'].replace('.pt', '')  # Remove '.pt'
    #logger.info(f"caption: {caption}")
    first_part, second_part = caption.rsplit('_', 1)  # Split at last underscore
    #logger.info(f"first_part: {first_part}, second_part: {second_part}")
    second_part = int(second_part)-1;
    #logger.info(f"second_part: {second_part}")
    logger.info(f"{i+1}.{captions[first_part][second_part]}, Score: {item['score']:.4f}")
  logger.info(f"completed")

image_name="1026792563.jpg"
image_path = os.path.join(images_path, image_name)
imageToCaptions(image_path)