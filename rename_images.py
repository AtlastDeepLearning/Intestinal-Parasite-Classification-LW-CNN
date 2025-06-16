import json
import os
import shutil
from pathlib import Path
import argparse
from collections import defaultdict

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return None

def create_category_mapping(data):
    """Create mappings for image IDs, categories, and annotations."""
    # Create a mapping of image IDs to their file names
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # Create a mapping of category IDs to category names
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create a mapping of image IDs to their categories from annotations
    image_id_to_categories = defaultdict(list)
    if 'annotations' in data:
        for ann in data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            if category_id in category_id_to_name:
                image_id_to_categories[image_id].append(category_id_to_name[category_id])
    
    return image_id_to_filename, category_id_to_name, image_id_to_categories

def rename_images(json_file, image_folder, output_folder=None):
    """Rename images based on their categories from the JSON file."""
    # Load JSON data
    data = load_json_file(json_file)
    if not data:
        return
    
    # Create category mappings
    image_id_to_filename, category_id_to_name, image_id_to_categories = create_category_mapping(data)
    
    # If no output folder specified, use the input folder
    if output_folder is None:
        output_folder = image_folder
    else:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
    
    # Process each image
    for img in data['images']:
        image_id = img['id']
        original_filename = img['file_name']
        
        # Get categories for this image
        categories = image_id_to_categories[image_id]
        
        if categories:
            # Join multiple categories with underscore if an image has multiple categories
            category_prefix = '_'.join(categories)
            # Create new filename: category_name_original_filename
            new_filename = f"{category_prefix}_{original_filename}"
            
            # Full paths
            original_path = os.path.join(image_folder, original_filename)
            new_path = os.path.join(output_folder, new_filename)
            
            try:
                if os.path.exists(original_path):
                    # Copy the file to the new location with the new name
                    shutil.copy2(original_path, new_path)
                    print(f"Renamed: {original_filename} -> {new_filename}")
                else:
                    print(f"Warning: Image file not found: {original_filename}")
            except Exception as e:
                print(f"Error processing {original_filename}: {e}")
        else:
            print(f"Warning: No categories found for image {original_filename}")

def main():
    parser = argparse.ArgumentParser(description='Rename images based on categories from a JSON file.')
    parser.add_argument('json_file', help='Path to the JSON file containing image categories')
    parser.add_argument('image_folder', help='Path to the folder containing the images')
    parser.add_argument('--output', '-o', help='Path to the output folder (optional)')
    
    args = parser.parse_args()
    
    # Check if JSON file exists
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file '{args.json_file}' does not exist.")
        return
    
    # Check if image folder exists
    if not os.path.exists(args.image_folder):
        print(f"Error: Image folder '{args.image_folder}' does not exist.")
        return
    
    # Rename images
    rename_images(args.json_file, args.image_folder, args.output)

if __name__ == "__main__":
    main() 