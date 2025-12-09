# Brain Tumor CT-Scan / X-Ray Classifier

This is a **brain tumor CT-scan classifier** designed to classify brain conditions from CT-scan or X-ray images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **Healthy (No Tumor)**

## How to Use:

1. **Upload an image**: Select a `.jpg` or `.png` file using the "Browse Files" button on the left.
2. **Click "Predict"**: Once the button turns green, the image is uploaded.
3. **View the result**: After processing, the model will classify the image and show the prediction with:
   - The identified tumor type (or healthy).
   - A brief description of the tumor type.
   - A graph displaying the probability distribution of each class.

## Architecture and Workflow:

- **Libraries Used**: Python 3.12.x, TensorFlow, Keras, NumPy, Streamlit (web interface), etc.
- **Model Choices**: VGG16, ResNet50V2, MobileNetV2
- **Dataset**: Brain Tumor MRI Dataset (Public - C00 License)
- **Development Environment**: Jupyter Notebooks
