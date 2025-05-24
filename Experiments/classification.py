import os
import sys
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping # type: ignore
import matplotlib.pyplot as plt

from Tools.metrics import MacroMetrics
from Models.pretrained import efficientnet_model, resnet_model
from Models.CNN import CNN_Model, CNN_V1, CNN_V2
from Tools.callbacks import MacroF1Callback, SaveBestModel, EarlyStoppingOnF1, ReduceLROnF1
from Utils import utils

class Experiment_Classification():
    def __init__(self,
                 role = 'articleType',
                 epochs=10,
                 batch_size=32,
                 learning_rate=0.0001,
                 target_size=None,
                 network_structure='CNN',
                 perform_augmentation = False,
                 base_project_dir='.'):
        
        self.role = role
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_size = target_size
        self.network_structure = network_structure
        self.perform_augmentation = perform_augmentation
        self.base_project_dir = base_project_dir
        self.task = self.role + '_classification_' + self.network_structure

        self.output_dir = os.path.join(self.base_project_dir, 'Output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.output_class_dir = os.path.join(self.output_dir, self.task)
        if not os.path.exists(self.output_class_dir):
            os.mkdir(self.output_class_dir)            

        self.output_models_dir = os.path.join(
            self.base_project_dir, 'Output_Models')
        if not os.path.exists(self.output_models_dir):
            os.mkdir(self.output_models_dir)

        self.output_models_dir = os.path.join(
            self.output_models_dir, self.task)
        if not os.path.exists(self.output_models_dir):
            os.mkdir(self.output_models_dir)

        # self.logs_dir = os.path.join(self.base_project_dir, 'logs')
        # if not os.path.exists("logs"):
        #     os.makedirs("logs")


        print(self.output_dir, self.output_models_dir)
        print("\n---------------------------------------------------")

        # Set the random seed for reproducibility
        utils.set_seed(42)

        # self.fold_history = FoldsHistory()
        self.prepare()

        if self.network_structure == 'CNN':
            self.model = CNN_Model(input_shape=self.input_shape, num_classes=self.num_classes)
        elif self.network_structure == 'CNN_V1':
            self.model = CNN_V1(input_shape=self.input_shape, num_classes=self.num_classes)
        elif self.network_structure == 'CNN_V2':
            self.model = CNN_V2(input_shape=self.input_shape, num_classes=self.num_classes)
        elif self.network_structure == 'efficientnet':
            self.model = efficientnet_model(input_shape=self.input_shape, num_classes=self.num_classes)
        elif self.network_structure == 'resnet':
            self.model = resnet_model(input_shape=self.input_shape, num_classes=self.num_classes, pr_dir=self.base_project_dir)

        self.model.summary(expand_nested=True)
        self.time = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")

        checkpoint_path = os.path.join(self.output_models_dir, "best_model.keras")
        # self.checkpoint = [
        #     ModelCheckpoint(
        #     filepath=checkpoint_path,  
        #     monitor='val_macro_f1',  # Monitoring F1-Score
        #     mode='max',  # Maximize F1-Score
        #     save_best_only=True,  # Only save the best model
        #     save_weights_only=False,  # Save entire model
        #     verbose=2
        #     ),
        #     ReduceLROnPlateau(
        #         monitor='val_macro_f1',
        #         factor=0.2,
        #         patience=2,
        #         mode='max',
        #         min_lr=1e-7,
        #         verbose=1
        #     ),
        #     EarlyStopping(
        #         monitor='val_macro_f1',
        #         patience=5,
        #         mode='max',
        #         restore_best_weights=True,
        #         verbose=1
        #     )
        # ]

        # Definition of F1, calculation callback
        macro_f1_callback = MacroF1Callback(
            validation_data=self.val_flow,
            num_classes=self.num_classes
        )

        save_best_model = SaveBestModel(save_path=checkpoint_path)
        early_stopping = EarlyStoppingOnF1(patience=5)
        reduce_lr = ReduceLROnF1(factor=0.2, patience=2)

        self.callbacks = [
            macro_f1_callback,
            save_best_model,
            early_stopping,
            reduce_lr
        ]


    def __create_model(self, decay=1e-6):
        # Initiate RMSprop optimizer
        # RMS_opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)
        opt = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, 
            beta_1=0.9,   
            beta_2=0.999
            )
        model = tf.keras.models.clone_model(self.model)
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=[
                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    # MacroMetrics(num_classes=self.num_classes)
            ]
        )
        return model


    def prepare(self):

        self.df = utils.load_dataset(base_dir=self.base_project_dir, role=self.role)

        # Filter Classes
        if self.role == 'articleType':
            self.df = utils.filter_classes_by_min_samples(self.df, class_column=self.role, min_samples=200)
        elif self.role == 'masterCategory':
            self.df = utils.filter_classes_by_min_samples(self.df, class_column=self.role, min_samples=50)
        elif self.role == 'subCategory':
            self.df = utils.filter_classes_by_min_samples(self.df, class_column=self.role, min_samples=100)     
        elif self.role == 'baseColour':
            self.df = utils.color_mapping(self.df, class_column=self.role)               
        elif self.role == 'usage':
            self.df = utils.filter_classes_by_min_samples(self.df, class_column=self.role, min_samples=100) 
        else:
            # Print summary statistics
            print(f"[{self.role}] column.")
            # class_counts = self.df[self.role].value_counts()
            print(f"Original number of classes: {len(self.df[self.role].value_counts())}")
            print(f"Original dataset size: {len(self.df)}")
            print("---------------------------------------------------")
        
        self.train_df, self.val_df = utils.custom_train_test_split(self.df, target_column=self.role, test_size=0.2, random_state=42)

        if self.perform_augmentation:
            utils.augment_minority_classes(train_df=self.train_df, role=self.role, base_dir=self.base_project_dir,  threshold_percent=9.5, target_size=self.target_size)

        self.train_flow, self.val_flow, self.class_indices = utils.preprocess_dataset( self.train_df,
                                                                                       self.val_df, 
                                                                                       model_type=self.network_structure, 
                                                                                       batch_size=self.batch_size, 
                                                                                       target_size=self.target_size,
                                                                                       base_dir=self.base_project_dir, 
                                                                                       role=self.role)

        
        # Reviewing input samples
        print("\n Check image preprocessing:")
        
        # Convert Dataset to Iterator and get a Batch
        train_iterator = iter(self.train_flow)
        x, y = next(train_iterator)

        # Checking pixel ranges after preprocessing
        print("Pixel range after preprocessing:", x.numpy().min(), x.numpy().max())

        # Checking the data format
        print("Data format - Images:", x.shape)  # It should be (batch_size, 128, 128, 3)
        print("Data format - Labels:", y.shape)  # Must be (batch_size, num_classes)

        self.num_classes = len(self.class_indices)
        # Receive a batch of data
        images, _ = next(train_iterator)
        # Check the dimensions of the first image
        self.input_shape = images.shape[1:]
        print("Input shape:", self.input_shape)

        # Show a label
        print("Label example (one-hot):", y.numpy()[0])
        print("Class index:", tf.argmax(y[0]).numpy())
        index_to_class = {v: k for k, v in self.class_indices.items()}
        predicted_index = tf.argmax(y[0]).numpy()
        print("Class name:", index_to_class[predicted_index])

        # # Show a sample image
        # plt.imshow((x[0] + 1)/2)  #Revert to the range [0,1] for display
        # plt.title(f"Label: {y[0]}")
        # plt.axis('off')
        # plt.show()
        # ===================================


        self.class_weights = utils.calculate_class_weights(dataframe=self.train_df, target_column=self.role)


    def train(self):

        self.history = None  # For recording the history of trainning process.
        print("\n---------------------------------------------------")
        model = self.__create_model()
        self.history = model.fit(
            self.train_flow,
            validation_data=self.val_flow,
            epochs=self.epochs,
            callbacks=self.callbacks,
            class_weight=self.class_weights  # Applying class weights
            )
        
        utils.save_training_results(self.history, output_dir=self.output_class_dir, time_stamp=self.time)