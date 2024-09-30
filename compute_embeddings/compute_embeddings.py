import json
import torch
from PIL import Image
import torchvision.transforms as T
import os
import gzip
import pickle

output_file = "embeddings"
batch_size = 400  # Adjust based on your GPU memory
mnist_dataset = False # Set to True if you are using the mnist6k dataset
folder_path_aerial = "aerial_images/" # Directory where the images are stored

# Specify here which transformations will be applied to the image
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Prepare a list with all images filenames to run everything in a loop fashion
transformed_images = [] # Initialize a list for the transformed images
image_id = []

# And here comes the loop iterating over all aerial images
for img_filename in os.listdir(folder_path_aerial):
    img_path = os.path.join(folder_path_aerial, img_filename) # Combine path and image name
    #print(img_path) # Let's print out the processed images
    if mnist_dataset:
        img = Image.open(img_path).convert('RGB') # The image is opened by PIL Image function (Think of PIL as a translator)
    else:
        img = Image.open(img_path) # The image is opened by PIL Image function
    t_img = transform(img) # The transformation is now applied here
    transformed_images.append(t_img)  # Append the transformed image to the list
    image_id_string = os.path.join(folder_path_aerial, img_filename)
    image_id.append(image_id_string)
# Convert list of tensors to a single 4D tensor (Recall: one image would be a 3D tensor)
tensor_aerial_images = torch.stack(transformed_images)

# tensor_images is now a 4D tensor of shape [N, C, H, W], where N is the number of images
print(tensor_aerial_images.shape)

# Load DINOv2 model
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move the model to GPU
dinov2_vits14_reg = dinov2_vits14_reg.to(device)

# Assuming tensor_aerial_images is a large tensor or list of tensors
num_images = len(tensor_aerial_images)

# Create an empty list to store the embeddings
aerial_embed_list = []

# Process images in batches
with torch.no_grad():  # Disable gradient computation
    for i in range(0, num_images, batch_size):
        # Create a batch of images (move to GPU if not already)
        batch = tensor_aerial_images[i:i + batch_size].to(device)
        
        # Forward pass to compute embeddings
        batch_embeddings = dinov2_vits14_reg(batch)
        
        # Move the computed embeddings to CPU to save GPU memory
        aerial_embed_list.append(batch_embeddings.cpu())
        
# Concatenate all the embeddings after processing
aerial_embed = torch.cat(aerial_embed_list, dim=0)

# Serialize with pickle
serialized_embeddings = pickle.dumps(aerial_embed)

# Compress the serialized data using gzip
compressed_embeddings = gzip.compress(serialized_embeddings)

output_dir = "computed_embeddings/"
os.makedirs(output_dir, exist_ok=True)

# Store the compressed data to a file
with open(output_dir + output_file + '.pkl.gz', 'wb') as f:
    f.write(compressed_embeddings)

# Store the image ids to a file
with open(output_dir + output_file + '_ids.json', 'w') as f:
    json.dump(image_id, f)