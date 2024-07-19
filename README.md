# Pottery Machine Learning Project

This project was developed as a way for me to expand my skills in machine learning, especially how to train models with limited data. The project is designed to take images of pottery at arbitrary rotations and output the image correctly oriented. The project is broken into several parts: Data collection and augmentation, model training, and use of the model.  

## Data Collection and Augmentation
Once the small dataset had been collected (around 100 images), I wrote a script to augment the data. The script takes in a directory of right-side-up images and creates numerous other images, using different cropping and lighting, then rotates the images 90 degrees at a time to create 4 folders of images at different rotations. This script also handles preprocessing the images, resizing them to 128x128 pixels. 

## Model Training
The model is trained within a docker container with a tensorflow image, and utilized my machine's GPU (A pain to set up I must say) to speed up training. The model is a Convolutional Neural Network (CNN) with 3 convolutional layers and 3 Dense layers. The model is trained on the augmented data and saved to a file.

## Use of the Model
Once trained, the model can be used to predict the orientation of new images with a reasonable degree of accuracy. main.py is a script that takes in a directory of images and flips them until they are right-side-up. 

