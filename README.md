# cat_dog_classifier
A model which can predict the given picture is of dog or a cat 
This model is based on dataset from kaggle with pre splitted folders of train , validation and test

# Dataset Source
Dataset: **Dog vs Cat FastAI**  
Provider: `arpitjain007/dog-vs-cat-fastai`

The dataset contains:
- `train/` → training images (cats & dogs)
- `valid/` → validation images (cats & dogs)
- `test1/` → unlabeled test images

The images of training set is augmented for the model

# Model Architecture
Custom CNN designed:

- 4 Convolutional Blocks (32 → 64 → 128 → 256  filters)
- MaxPooling after each convolution block
- BatchNormalization + Dropout to prevent overfitting
- GlobalAveragePooling2D instead of Flatten
- Dense Layer (512 neurons)
- Final output: *sigmoid* for binary classification

#  Necessities for training the model
- Image Size --> 256 × 256 
- Batch Size --> 32 
- Loss Function --> Binary Crossentropy (only two classes to classify)
- Optimizer --> adam
- Epochs --> 30 (EarlyStopping prevents overfitting) 
- Metrics --> Accuracy 

 # Result and Evaluation metrices
 The model outputs:
- Training & Validation Accuracy Graph  
- Training & Validation Loss Graph  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-Score)

