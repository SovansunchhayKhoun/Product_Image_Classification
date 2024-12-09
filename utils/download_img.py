import os
import requests

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
