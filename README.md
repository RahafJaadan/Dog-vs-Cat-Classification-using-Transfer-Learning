# Cat vs Dog Image Classifier 
This project aims to develop an AI model that predicts whether an image contains a cat or a dog. The model was built using deep learning, with a focus on improving accuracy and addressing the overfitting issue

## Description 
We trained the model using data from 'Kaggle' with images labeled into categories (cats and dogs). We encountered challenges in balancing the model’s accuracy between training and testing data. To overcome the overfitting problem, we used Pre-Trained Models such as VGG16 and MobileNetV2. 

## Requirements
* Python
* Libraries: TensorFlow, NumPy, OpenCV, scikit-learn, Matplotlib, PIL

## Usage
1. Install the requirements:
```
   pip install tensorflow numpy opencv-python scikit-learn matplotlib pillow
```
2. Download training data from Kaggle.
3. Run the model:
  ```
   python train_model.py
```
4. Use the model to classify images:
  ```
   from model import predict
 result = predict("path/to/image.jpg")
```

## Techniques Used 
* **VGG16** and **MobileNetV2** as Pre-Trained Models to improve accuracy.
* Image processing using OpenCV. 
* Data splitting and model performance validation with scikit-learn.
