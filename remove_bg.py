from PIL import Image
import os

def remove_background(image_path, mask_path, output_path):
    try:
        image = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        # Resize mask to image size
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)

        # Apply mask as alpha channel
        image.putalpha(mask)

        image.save(output_path)
        print(f"Background removed and saved to: {output_path}")

    except FileNotFoundError:
        print("Error: One or both of the image or mask files were not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
image_path = "U-2-Net/test_data/test_images/0003.jpg"  # Replace with your image path
mask_path = "U-2-Net/test_data/u2net_results/0003.png"  # Replace with your mask path
output_path = "U-2-Net/test_data/removed_background/0003_removed.png" #output path.

#create the output directory if it does not exist.
os.makedirs(os.path.dirname(output_path), exist_ok=True)

remove_background(image_path, mask_path, output_path)

#Example for processing all files.
# image_dir = "imgs_test/"
# mask_dir = "mask_test/"
# output_dir = "U-2-Net/test_data/removed_background/"
image_dir = "U2Net/test_data/test_my_images/"
mask_dir = "U2Net/test_data/u2net_results/"
output_dir = "U2Net/test_data/removed_background/"

os.makedirs(output_dir, exist_ok=True)

image_files = os.listdir(image_dir)

for image_file in image_files:
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, os.path.splitext(image_file)[0] + ".png")
        output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_removed.png")
        if os.path.exists(mask_path):
            remove_background(image_path, mask_path, output_path)
        else:
            print(f"Mask file not found for: {image_file}")