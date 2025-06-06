# Fashion Product Classification

This project implements deep learning models to classify fashion product images across multiple categories such as article type, master category, sub-category, color, season, and usage.

## Project Structure

```
.
├── Utils
│ ├── utils.py                # Utility functions for data loading, preprocessing, and augmentation
│
├── Models
│ ├── CNN.py                  # Custom CNN architectures (CNN_Model, CNN_V1, CNN_V2)
│ │              
│ ├── pretrained.py           # Pretrained models (EfficientNetB4, ResNet50)
│
├── Experiments
│ ├── classification.py       # Main experiment class for training and evaluation
│
├── Tools
│ ├── callbacks.py            # Custom training callbacks (F1 monitoring, early stopping)
│
├── annotated_images          # Images with classification and caption annotations
|
├── Output                    # Stores training logs and results
│
├── Output_Models             # Stores saved models
|
├── caption_generation.py     # geneate caption & predict images
│
└── Experiment.py             # Main script to run experiments
```


## Dataset

The project uses the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle, which contains:

- 44,000+ fashion product images
- Multiple classification categories:
  - gender
  - masterCategory (e.g., Apparel, Footwear)
  - subCategory (e.g., Tops, Shoes)
  - articleType (e.g., Tshirts, Jeans)
  - baseColour
  - season
  - usage

## Features

1. **Multiple Model Architectures**:
   - Custom CNNs (3 variants with increasing complexity)
   - Pretrained models (EfficientNetB4, ResNet50)

2. **Advanced Training Features**:
   - Class weighting for imbalanced data
   - Data augmentation for minority classes
   - Custom callbacks for F1-score monitoring
   - Learning rate scheduling based on validation performance

3. **Comprehensive Evaluation**:
   - Tracks precision, recall, and F1-score (macro-averaged)
   - Visualizes training history (accuracy/loss curves)
   - Saves best model based on validation F1-score

4. **Image Captioning**:
   - Generates descriptive captions using BLIP model
   - Combines classification results with generative AI
   - Creates annotated images with classification results and captions

## Installation

1. Download the dataset from Kaggle and place it in the `Data/` directory:
```
Data/
├── resized_images/        # Contains all product images
└── styles.csv             # Contains product metadata
```

## Usage

Run the main experiment script:
```bash
python Experiment.py
```

### Image Caption Generation

Generate captions and annotated images:
```bash
python caption_generation.py
```

Example output images will be saved in `annotated_images/`

### Example Annotated Image

<img src="annotated_images/1533_annotated.jpg" alt="Fashion Product" width="400" height="400">
*Caption: A stylish white men caps from nike*

### Customizing Experiments

- You can modify Experiment.py to:
   - Change target classification category (`role`)
   - Adjust training parameters (epochs, learning rate)
   - Switch model architecture (`network_structure`)
   - Enable/disable data augmentation

- Available model architectures:
   - `CNN` - Basic CNN
   - `CNN_V1` - CNN with BatchNorm
   - `CNN_V2` - CNN with BatchNorm and Dropout
   - `efficientnet` - EfficientNetB4
   - `resnet` - ResNet50

## Results

- The training process generates:
   - Model checkpoints (saved in `Output_Models/`)
   - Training history plots (accuracy/loss)
   - CSV files with metrics
   - Augmentation reports (when enabled)

## Key Implementation Details

1. **Data Preprocessing**:
   - Automatic class filtering (minimum samples per class)
   - Smart color grouping (e.g., "Navy Blue" → "Blue")
   - Custom image preprocessing for each model type

2. **Class Imbalance Handling**:
   - Automatic class weighting
   - Smart augmentation targeting minority classes

3. **Training Optimization**:
   - Custom F1-score callback
   - Early stopping based on validation F1
   - Learning rate reduction on plateau