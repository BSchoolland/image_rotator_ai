from PIL import Image, ImageEnhance, ExifTags
import numpy as np
import os
from tqdm import tqdm


IMG_SIZE = 128

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def crop_to_square(image):
    width, height = image.size
    min_dimension = min(width, height)
    left = (width - min_dimension) / 2
    top = (height - min_dimension) / 2
    right = (width + min_dimension) / 2
    bottom = (height + min_dimension) / 2
    return image.crop((left, top, right, bottom))

def transpose_image(image, angle):
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_270)
    elif angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    elif angle == 270:
        return image.transpose(Image.Transpose.ROTATE_90)
    return image


def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    return image

RANDOM_CROP_PIXELS = 20

def generate_permutations(input_dir, output_base_dir, num_lightning_variations=5, num_crop_variations=5):
    # Ensure output directories exist
    rotations = [0, 90, 180, 270]
    for rotation in rotations:
        output_dir = os.path.join(output_base_dir, f"{rotation}_degrees")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    image_count = 0
    # Process each image in the input directory
    for image_name in tqdm(os.listdir(input_dir), desc="Processing Images"):
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path)
        image = correct_image_orientation(image)
        # Crop to square
        cropped_image = crop_to_square(image)

        cropped_image = cropped_image.resize((IMG_SIZE + RANDOM_CROP_PIXELS * 2, IMG_SIZE + RANDOM_CROP_PIXELS * 2))
        # random crop
        for i in range(num_crop_variations):
            left = np.random.randint(0, RANDOM_CROP_PIXELS * 2)
            top = np.random.randint(0, RANDOM_CROP_PIXELS * 2)
            right = left + IMG_SIZE
            bottom = top + IMG_SIZE
            rand_cropped_image = cropped_image.crop((left, top, right, bottom))
            for rotation in rotations:
                transposed_image = transpose_image(rand_cropped_image, rotation)

                output_dir = os.path.join(output_base_dir, f"{rotation}_degrees")
                transposed_image.save(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_crop_{i}_rotation_{rotation}.png"))
                image_count += 1
                for ii in range(num_lightning_variations):
                    brightness_factor = np.random.uniform(0.5, 1.5)
                    brightened_image = adjust_brightness(transposed_image, brightness_factor)
                    brightened_image.save(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_crop_{i}_rotation_{rotation}_lightning_{ii}.png"))
                    image_count += 1
    print(f'Generated {image_count} images from a starting set of {len(os.listdir(input_dir))} images')

# Example usage
input_dir = 'test-code/input'
output_base_dir = 'test-code/output'
generate_permutations(input_dir, output_base_dir)
