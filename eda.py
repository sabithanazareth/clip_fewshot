import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def visualise(train_dir):
# List the subfolders in the 'train' directory, which represent sleeve types
  sleeve_types = os.listdir(train_dir)

  # Define the number of sample images to visualize for each class
  num_samples_per_class = 6

  # Create a grid of subplots to display sample images
  fig, axs = plt.subplots(len(sleeve_types), num_samples_per_class, figsize=(15, 15))

  # Loop through each sleeve type and visualize sample images from the 'train' directory
  for i, sleeve_type in enumerate(sleeve_types):
      sleeve_images = os.path.join(train_dir, sleeve_type)
      image_files = os.listdir(sleeve_images)
      selected_images = random.sample(image_files, num_samples_per_class)
      
      for j, image_filename in enumerate(selected_images):
          image_path = os.path.join(sleeve_images, image_filename)
          img_train = Image.open(image_path)
          axs[i, j].imshow(img_train)
          axs[i, j].set_title(sleeve_type)

  # Set axis labels and display the plot
  for ax in axs.flat:
      ax.axis('off')

  plt.show()