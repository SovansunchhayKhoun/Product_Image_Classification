from PIL import Image
import shutil
import os
import requests
import numpy as np
from scipy.ndimage import median_filter

def convert_images_to_jpg(input_dir, output_dir):
    """
    Converts all images in the input directory to .jpg format 
    and saves them to the output directory.

    Parameters:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory to save converted .jpg images.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for i, filename in enumerate(os.listdir(input_dir)):
        print(f"{i}/{len(os.listdir(input_dir))}")
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Open the image
            with Image.open(input_path) as img:
                # Convert to RGB (to avoid issues with grayscale or RGBA images)
                img = img.convert("RGB")
                # Save as .jpg
                img.save(output_path, format="JPEG")  # Adjust quality if needed
            print(f"Converted {filename} to {output_filename}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

def move_files(source_file, target_dir):
    # Check if the source file exists
    if not os.path.isfile(source_file):
        print(f"Error: Source file '{source_file}' does not exist.")
        return
    
    # Ensure the target directory exists, if not, create it
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Target directory '{target_dir}' created.")
    
    # Get the filename from the source file path
    filename = os.path.basename(source_file)
    
    # Define the target file path
    target_file = os.path.join(target_dir, filename)
    
    try:
        # Move the file
        shutil.move(source_file, target_file)
        print(f"Moved: {filename} to {target_dir}")
    except Exception as e:
        print(f"Error moving {filename}: {e}")

def download_images(image_urls, save_path):
    """
    Download images from a list of URLs and save them to a specified path.

    Parameters:
    - image_urls (list): List of image URLs to download.
    - save_path (str): Directory where images will be saved.

    Returns:
    - None
    """
    # Ensure the save_path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, url in enumerate(image_urls):
        try:
            print(f"Downloading image {i + 1}/{len(image_urls)}: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            
            # Extract the file name from the URL
            file_name = os.path.basename(url.split("?")[0])  # Remove query parameters if any
            file_path = os.path.join(save_path, file_name)

            # Save the image
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            print(f"Saved to {file_path}")
        except Exception as e:
            print(f"Failed to download {url}. Error: {e}")

def resize_img_and_noise_filter(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for i, filename in enumerate(os.listdir(input_dir)):
        print(f"{i}/{len(os.listdir(input_dir))} -> Resizing {input_dir+filename}")
        # Open the image
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        
        # Resize the image to 300x300
        img_resized = img.resize((300, 300))
        
        # Convert the image to a NumPy array
        img_array = np.array(img_resized)
        
        # Apply the median filter to each channel separately
        if len(img_array.shape) == 3:  # Check if the image has 3 channels
            filtered_array = np.zeros_like(img_array)
            for channel in range(3):  # Assuming 3 channels (R, G, B)
                filtered_array[:, :, channel] = median_filter(img_array[:, :, channel], size=3)
        else:  # For grayscale images
            filtered_array = median_filter(img_array, size=3)
        
        # Convert the filtered array back to an image
        img_filtered = Image.fromarray(filtered_array)
        
        # Save the processed image in the output directory
        output_path = os.path.join(output_dir, filename)
        img_filtered.save(output_path)

    print(f"All images have been processed and saved to {output_dir}.")
