import os
import shutil

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
