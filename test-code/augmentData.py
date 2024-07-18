from PIL import Image, ImageEnhance, ExifTags
import numpy as np
import os

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
    # if angle == 90:
    #     return image.transpose(Image.Transpose.ROTATE_270)
    # elif angle == 180:
    #     return image.transpose(Image.Transpose.ROTATE_180)
    # elif angle == 270:
    #     return image.transpose(Image.Transpose.ROTATE_90)
    return image


def random_crop(image, crop_size_percent):
    width, height = image.size
    crop_width = int(width * crop_size_percent)
    crop_height = int(height * crop_size_percent)
    left = np.random.randint(0, width - crop_width)
    top = np.random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

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

def generate_permutations(input_dir, output_base_dir, num_lightning_variations=5):
    # Ensure output directories exist
    rotations = [0, 90, 180, 270]
    for rotation in rotations:
        output_dir = os.path.join(output_base_dir, f"{rotation}_degrees")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Process each image in the input directory
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path)
        image = correct_image_orientation(image)
        # Crop to square
        cropped_image = crop_to_square(image)

        cropped_image = cropped_image.resize((IMG_SIZE, IMG_SIZE))  
        # rescale to a slightly smaller size to make future processing faster
        # as a test save the cropped image
        for rotation in rotations:
            transposed_image = transpose_image(cropped_image, rotation)
            output_dir = os.path.join(output_base_dir, f"{rotation}_degrees")
            transposed_image.save(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_rotation_{rotation}.png"))
            print('saved to', os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_rotation_{rotation}.png"))
            
            for i in range(num_lightning_variations):
                brightness_factor = np.random.uniform(0.5, 1.5)
                brightened_image = adjust_brightness(transposed_image, brightness_factor)
                brightened_image.save(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_rotation_{rotation}_bright_{i}.png"))

# Example usage
input_dir = 'test-code/input'
output_base_dir = 'test-code/output'
generate_permutations(input_dir, output_base_dir)
