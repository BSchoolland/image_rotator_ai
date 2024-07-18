import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt

# Define paths and parameters
inference_dir = 'inference'
img_height, img_width = 128, 128
# rotations: 0, 180, 270, 90 degrees
class_types = ['0_degrees', '180_degrees', '270_degrees', '90_degrees']

# corrected rotation output
output_dir = "output"

# Load the trained model
model = load_model('app/model.keras')

# Function to load and preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Get list of image files in the inference directory
image_files = [f for f in os.listdir(inference_dir) if os.path.isfile(os.path.join(inference_dir, f))]

# Run inference on each image
for image_file in image_files:
    image_path = os.path.join(inference_dir, image_file)
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_types[predicted_class]
    
    # Display the result with confidence levels
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}\n' +
              '\n'.join([f'{cls}: {conf:.2f}' for cls, conf in zip(class_types, predictions[0])]))
    plt.axis('off')
    plt.show()
    # rotate the image to be right side up
    if predicted_class == 1:
        img = np.rot90(img, 2)
    elif predicted_class == 2:
        img = np.rot90(img, 3)
    elif predicted_class == 3:
        img = np.rot90(img, 1)
    plt.imshow(img)
    # save the corrected image
    plt.imsave(os.path.join(output_dir, image_file), img)
    
