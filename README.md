# Sign Language Recognition Using Convolutional Neural Networks (CNN)
# 1. Introduction
Sign language is an essential means of communication for individuals with hearing and speech impairments. However, the lack of fluency in sign language by the general public creates a communication barrier. This project aims to bridge this gap by developing a deep learning-based system that can recognize American Sign Language (ASL) signs from images and convert them into their corresponding alphabets. The solution is built using Convolutional Neural Networks (CNN) due to their success in image classification tasks.

# 2. Problem Statement
The main objective of this project is to develop a model that can accurately classify images of hands displaying different ASL letters. The model will be trained to recognize 26 distinct hand gestures representing the English alphabet, enabling the translation of ASL signs into text.

# 3. Dataset
The dataset used for this project contains images of hand gestures representing different ASL alphabets. These images are stored in folders, each labeled according to the respective sign (A-Z). The images are resized to a uniform size of 128x128 pixels for consistency. The dataset is divided into training, validation, and testing sets as follows:

Training set: 2015 images
Validation set: 252 images
Testing set: 252 images


# 4. Methodology
4.1 Data Preprocessing
The dataset was processed as follows:

Images were loaded using the OpenCV library.
They were resized to 128x128 pixels to standardize input size.
Converted from BGR to RGB for correct color representation.
Labels were extracted from folder names representing each sign.

4.2 Data Split
The dataset was divided into three parts:

Training set (80%): Used for model training.
Validation set (10%): Used for hyperparameter tuning and validation.
Test set (10%): Used for final evaluation.

4.3 Model Architecture
A Convolutional Neural Network (CNN) was designed with the following layers:

Convolutional Layers: Extract features from images by applying filters.
Pooling Layers: Reduce the spatial dimensions of the feature maps.
Dropout Layers: Prevent overfitting by randomly setting a fraction of input units to zero during training.
Flatten Layer: Converts the 2D matrix data into a 1D vector.
Dense (Fully Connected) Layers: Classifies the features into different categories.

4.4 Training the Model
The model was compiled using the following configurations:

Loss function: 
Categorical cross-entropy (for multi-class classification).
Optimizer: 
Adam optimizer, which adapts the learning rate during training.
Metrics: 
Accuracy was used as the primary metric to evaluate the model's performance.

The training process involved splitting the dataset and running the model over multiple epochs to optimize weights and reduce loss.

# 5. Results
After training the model, the following performance metrics were computed:
Overall Accuracy: The model has an overall accuracy of 93%, meaning it correctly classified 93% of the total instances (252 samples).

Macro Average:
* Precision: 0.93
* Recall: 0.92
* F1-Score: 0.92
The macro average gives an unweighted average across all classes, treating each class equally.

Weighted Average:

* Precision: 0.94
* Recall: 0.93
* F1-Score: 0.93
The weighted average takes into account the support (number of instances) of each class, so classes with more instances have more influence on the final score.

![image](https://github.com/user-attachments/assets/207811c2-c82b-4633-a7be-c3d24feb0151)


# 6. Conclusion
This project successfully developed a CNN-based model for recognizing American Sign Language hand gestures. The model achieved significant accuracy on the test set, indicating its potential for real-world applications. Future improvements could include expanding the dataset, using data augmentation, and developing a real-time detection system.
