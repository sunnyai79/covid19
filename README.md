# COVID-19 Detection using X-Ray Images
This project aims to detect COVID-19 in patients using X-ray images by implementing deep learning techniques. The goal is to classify images into four categories: **COVID**, **Lung Opacity**, **Normal**, and **Viral Pneumonia** using a transfer learning approach with the **VGG19** model.

### Problem Statement
The primary objective is to identify whether a patient has been diagnosed with COVID-19 by analyzing X-ray images using effective deep learning models.

### Dataset
The dataset used for this project can be found on Kaggle:
[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### Libraries Used
TensorFlow
Keras
Pandas
Numpy
Matplotlib
Scikit-learn

### Model Architecture
The project leverages the **VGG19** architecture, pre-trained on ImageNet, as a base model. A custom head is added with dense, dropout, and softmax layers to fine-tune the model for the COVID-19 detection task. The model is trained using a categorical cross-entropy loss function and Adam optimizer.

### Steps:
1. **Data Preprocessing**:  
   - Images are resized to 224x224, normalized, and augmented using the `ImageDataGenerator`.
   
2. **Exploratory Data Analysis**:  
   - Plots of sample images from each class and distribution of images per class are created for visualization.
   
3. **Model Training**:  
   - The VGG19 model is fine-tuned, freezing the base layers and training custom layers.
   
4. **Evaluation**:  
   - The model is evaluated based on accuracy and loss curves.
   - Confusion matrix and classification report provide insight into class-level performance.

### Results
The model achieved an overall accuracy of **72%** on the validation set. However, further improvements can be made to the model by fine-tuning the learning rate, increasing the number of epochs, and experimenting with other models like **ResNet** or **Inception** for better results.

### Observations
- The model performs well but struggles with some specific class labels such as COVID.
- Improvements in precision and recall could be made with additional data and further tuning.

### Future Enhancements
- Fine-tune the base model layers for improved performance.
Increase the dataset size for more robust training.
Explore other model architectures like ResNet or Inception.
