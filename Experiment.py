import os
from Models.CNN import CNN_Model, CNN_V1
from Utils import utils
from Experiments.classification import Experiment_Classification

base_project_dir = os.path.abspath(os.path.dirname(__file__))

# List of columns to sort by  
roles = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season','usage']

# Building and training a model for each column
for role in roles:
    print(f"\n Start training the model for the column: {role}")
    
    model = Experiment_Classification(
        role=role,
        epochs=50,
        learning_rate=1e-5,
        target_size=(120, 120),
        network_structure='resnet',
        perform_augmentation=False,
        base_project_dir=base_project_dir
    )
    
    model.train()