# Sugarcane-Disease-Detection-Python
Sugarcane Disease Detection
Overview

This project focuses on detecting diseases in sugarcane crops using image processing and machine learning techniques. By leveraging computer vision algorithms and trained machine learning models, the system can identify and classify common diseases affecting sugarcane, such as Red Rot, Top Shoot Borer, and Rust. The goal is to provide farmers and agricultural specialists with an automated tool to help monitor the health of sugarcane crops and implement timely interventions.
Features

    Disease Detection: Identify various diseases in sugarcane using images of the leaves and stems.
    Model Training: Train machine learning models using labeled datasets of sugarcane plant images to enhance detection accuracy.
    Real-time Diagnosis: Provide real-time feedback on the health of sugarcane plants using uploaded images.

Prerequisites

Before running this project, ensure you have the following installed:

    Python 3.6 or higher
    Libraries:
        TensorFlow or Keras (for machine learning models)
        OpenCV (for image processing)
        scikit-learn (for model evaluation)
        Pandas (for handling data)
        NumPy (for numerical calculations)
        Matplotlib (for plotting images and results)

Install the required dependencies by running the following command:

pip install -r requirements.txt

Dataset

The dataset used for training the model consists of labeled images of sugarcane leaves and stems. The images are categorized based on the type of disease or healthy plant. Example diseases include:

    Red Rot
    Top Shoot Borer
    Rust

You can download the dataset from this link or upload your own set of labeled images for further training.
Getting Started

    Clone the repository:

git clone https://github.com/your-username/sugarcane-disease-detection.git
cd sugarcane-disease-detection

Preprocess the data: The dataset needs to be processed into a format suitable for machine learning. Run the following script to preprocess the images:

python preprocess_data.py

Train the model: After preprocessing, you can train the model using the following command:

python train_model.py

The training process will take time depending on the size of the dataset and the computational power of your system. Once the model is trained, the weights will be saved in the models/ directory.

Test the model: After training, you can test the model by running:

    python test_model.py --image path_to_image

    The script will classify the uploaded image and return the predicted disease type or indicate if the plant is healthy.

    Deploy for real-time predictions: You can deploy the model into a web or mobile application for real-time disease detection.

Model Architecture

The model used for sugarcane disease detection is based on Convolutional Neural Networks (CNNs), which are effective in image classification tasks. The architecture consists of several convolutional layers, pooling layers, and fully connected layers.
Evaluation

The performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1-score. The evaluation process can be done using the evaluate_model.py script.

python evaluate_model.py

Example Output

    Healthy plant: 95% confidence
    Red Rot: 85% confidence
    Top Shoot Borer: 75% confidence

Usage

To detect diseases in sugarcane, you can upload images of the sugarcane leaves/stems to the system, and it will predict if the plant is infected and which disease it has. This can assist in crop management, early intervention, and preventing further spread of diseases.
