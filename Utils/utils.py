import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input  # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import uuid

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across Python, NumPy, and TensorFlow.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '4' 
    os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
  

def load_dataset(base_dir=None, role=None):

    print(f"Load dataset...\n Column: {role}")

    image_path = os.path.join(base_dir, 'Data/resized_images/')
    csv_path = os.path.join(base_dir, 'Data/styles.csv')

    # Read CSV file
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Check if the specified role column exists
    if role not in df.columns:
        raise ValueError(f"Column '{role}' not found in the dataset. Available columns: {list(df.columns)}")
    
    # Drop rows with missing values in id or the specified role column
    df = df.dropna(subset=['id', role])

    # Convert id to image file names
    df['image'] = df['id'].astype(str) + '.jpg'

    # Keep only data that has a corresponding photo
    df = df[df['image'].isin(os.listdir(image_path))]

    # Select only the necessary columns
    df = df[['image', role]]
    df = df.reset_index(drop=True)

    # Print dataset information
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print("---------------------------------------------------")

    return df


def filter_classes_by_min_samples(df, class_column=None, min_samples=None):

    print("Filters out classes with fewer than the specified minimum number of samples...\n")
    # Calculate class frequencies
    class_counts = df[class_column].value_counts()

    # Identify classes meeting the minimum sample requirement
    valid_classes = class_counts[class_counts >= min_samples].index

    # Filter the original dataframe
    df_filtered = df[df[class_column].isin(valid_classes)]

    # Print summary statistics
    print(f"Filter [{class_column}] column.")
    print(f"Original number of classes: {len(class_counts)}")
    print(f"Number of classes retained: {len(valid_classes)}")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(df_filtered)}")
    print(f"Minimum samples per class: {min_samples}")
    print("---------------------------------------------------")

    return df_filtered


def custom_train_test_split(df, target_column=None, test_size=0.2, random_state=42):

    print("Separating data into training and validation Set...\n")
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Perform stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],  # Stratify based on target column
        random_state=random_state
    )
    
    # Print statistics
    print(f"Number of training set samples: {train_df.shape[0]}")
    print(f"Number of validation set samples: {val_df.shape[0]}")
    print(f"Split ratio: {1-test_size:.0%} train / {test_size:.0%} validation")
    print("---------------------------------------------------")
    
    return train_df, val_df
  
def augment_minority_classes(train_df, role, base_dir, threshold_percent=5.0, report_dir=None, target_size=None):
    """
    Augment classes in train_df that have fewer than threshold_percent of total samples.
    Now with built-in smart target calculation and unique augment file naming (.jpg).
    """

    # Set an increase goal based on the current number
    def get_target(count, target_role):
        if target_role == 'gender':
            if count < 500: return 1500    # Girls, Boys
            elif count < 1000: return 1200 # Boys
            elif count < 2500: return 2500 # Unisex
            else: return count             #Men, Women
        elif target_role == 'masterCategory':
            if count < 200: return 2000    # Free Items (105 → 2000)
            elif count < 2500: return 5000 # Personal Care (2403 → 5000)
            elif count < 10000: return 12000 # Footwear (9219 → 12000)
            elif count < 15000: return 15000 # Accessories (11274 → 15000)
            else: return count  # Apparel
        elif target_role == 'subCategory':
            if count < 50: return min(500, count*10)
            elif count < 200: return min(1000, count*5)
            elif count < 1000: return min(3000, count*2)
            elif count < 5000: return min(8000, int(count*1.5))
            else: return count
        elif target_role == 'baseColour':
            if count < 400: return 1500    # Multi, Gold
            elif count < 900: return 2000  # Orange, Yellow
            elif count < 1200: return 2500 # Silver
            elif count < 1800: return 3000 # Purple
            elif count < 2500: return 3500 # Pink
            elif count < 3500: return 4500 # Green, Red
            elif count < 5000: return 6000 # Grey, Brown
            else: return count             # Black, Blue, White
        elif target_role == 'season':
            if count < 3000: return 8000  # Spring
            elif count < 9000: return 12000  # Winter
            else: return count  # Fall, Summer
        elif target_role == 'usage':
            if count < 2500: return 5000    # Formal (2345 → 5000)
            elif count < 3500: return 6000  # Ethnic (3208 → 6000)
            elif count < 4500: return 8000  # Sports (4025 → 8000)
            else: return count              # Casual
        else:
            if count < 300: return 1200
            elif count < 600: return 900
            elif count < 1000: return 700
            else: return count

    print(f"\nAugmenting minority classes under {threshold_percent:.1f}% of training data...\n")

    source_dir = os.path.join(base_dir, 'Data', 'resized_images')
    target_dir = os.path.join(base_dir, 'Data', f'image_classification_{role}', 'train')
    
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f'augmentation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        report_data = []

    class_counts = train_df[role].value_counts()
    total_samples = len(train_df)
    threshold = total_samples * (threshold_percent / 100)
    minority_classes = class_counts[class_counts < threshold].index.tolist()
    print(f"Minority Classes: {minority_classes}\n")

    if role == 'baseColour':
        datagen = ImageDataGenerator(
            rotation_range=10,         
            width_shift_range=0.05,     
            height_shift_range=0.05,    
            zoom_range=0.05,            
            horizontal_flip=True,       
            fill_mode='nearest'         
        )
    elif role == 'usage':
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            brightness_range=(0.9, 1.1),
            horizontal_flip=True,
            zoom_range=0.05,
            fill_mode='reflect'
        )
    else:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    before_stats = class_counts.to_dict()
    
    for class_name in minority_classes:
        class_images = train_df[train_df[role] == class_name]['image'].tolist()
        existing_count = len(class_images)
        target_count = get_target(existing_count, target_role=role)
        aug_needed = max(0, target_count - existing_count)
        
        if aug_needed == 0:
            print(f"Class '{class_name}' has {existing_count} samples - no augmentation needed.")
            continue

        print(f"\nAugmenting class '{class_name}' ({existing_count} -> {target_count}): {aug_needed} new images needed...")

        dst_class_dir = os.path.join(target_dir, str(class_name))
        os.makedirs(dst_class_dir, exist_ok=True)

        per_image_aug = max(1, aug_needed // existing_count)
        extra_aug = aug_needed % existing_count
        augmented_count = 0

        for idx, image_name in enumerate(class_images):
            current_aug = per_image_aug + (1 if idx < extra_aug else 0)
            if current_aug <= 0:
                continue

            src = os.path.join(source_dir, image_name)
            try:
                img = load_img(src, target_size=target_size)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                for i in range(current_aug):
                    for batch in datagen.flow(x, batch_size=1):
                        unique_id = uuid.uuid4().hex[:6]  # Generate a unique identifier
                        prefix = os.path.splitext(os.path.basename(image_name))[0]
                        aug_filename = f"{prefix}_aug_{i}_{unique_id}.jpg"  # Save as .jpg format
                        batch_img = array_to_img(batch[0])
                        batch_img.save(os.path.join(dst_class_dir, aug_filename))
                        augmented_count += 1
                        break
            except Exception as e:
                print(f"❌ Error augmenting {image_name}: {e}")
                if report_dir:
                    report_data.append({
                        'class': class_name,
                        'original_image': image_name,
                        'augmented_count': 0,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })

        print(f"✅ Added {augmented_count} augmented images for class '{class_name}'")
        
        if report_dir:
            report_data.append({
                'class': class_name,
                'original_count': existing_count,
                'target_count': target_count,
                'augmented_count': augmented_count,
                'error': None,
                'timestamp': datetime.now()
            })

    if report_dir and report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_path, index=False)
        print(f"\n Augmentation report saved to: {report_path}")

        after_stats = {class_name: len(os.listdir(os.path.join(target_dir, str(class_name)))) 
                      for class_name in minority_classes}
        
        summary_df = pd.DataFrame({
            'Class': minority_classes,
            'Before_Augmentation': [before_stats.get(c, 0) for c in minority_classes],
            'After_Augmentation': [after_stats.get(c, before_stats.get(c, 0)) for c in minority_classes],
            'Increase': [after_stats.get(c, before_stats.get(c, 0)) - before_stats.get(c, 0) 
                        for c in minority_classes]
        })
        
        summary_path = os.path.join(report_dir, f'augmentation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f" Augmentation summary saved to: {summary_path}")

    print("\n✅ Augmentation of minority classes completed.")
    print("---------------------------------------------------\n")


def calculate_class_weights(dataframe, target_column=None):
    
    # Calculate class counts
    class_counts = dataframe[target_column].value_counts().sort_index()
    
    # Calculate total samples and number of classes
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    
    # Calculate weights using balanced method (inverse frequency)
    class_weights = {
        i: total_samples / (num_classes * count)
        for i, count in enumerate(class_counts)
    }
    
    print("Class Weights:", class_weights)
    return class_weights
    

def preprocess_dataset( train_df, 
                        val_df, 
                        model_type=None, 
                        batch_size=32, 
                        target_size=None,
                        base_dir=None, 
                        role=None):
    """
    Preprocess dataset using image_dataset_from_directory by creating class folders.
    """

    print("Preprocessing using image_dataset_from_directory...\n")

    # Define paths
    source_dir = os.path.join(base_dir, 'Data', 'resized_images')
    target_base_dir = os.path.join(base_dir, 'Data', 'image_classification_' + role)

    train_dir = os.path.join(target_base_dir, 'train')
    val_dir = os.path.join(target_base_dir, 'val')

    # Only create directories and copy data if they don't exist
    def prepare_split(df, split_dir):
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
            print(f"Created directory: {split_dir}")
            
            # Copy images to class folders
            for _, row in df.iterrows():
                class_name = str(row[role])
                image_name = row['image']
                src = os.path.join(source_dir, image_name)
                dst_dir = os.path.join(split_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, image_name)
                if os.path.exists(src):
                    shutil.copy(src, dst)
        else:
            print(f"Directory already exists: {split_dir} (Skipping copy)")

            
    # Copy train and val images
    prepare_split(train_df, train_dir)
    prepare_split(val_df, val_dir)

    def get_preprocess_fn(model_type):
        if model_type == 'efficientnet':
            print('Using [efficientnet] manual preprocess...')
            return lambda x: (x / 127.5) - 1.0
        elif model_type == 'resnet':
            print('Using [resnet] preprocess...')
            from tensorflow.keras.applications.resnet import preprocess_input # type: ignore
            return preprocess_input
        elif model_type == 'inception':
            print('Using [inception] preprocess...')
            from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore
            return preprocess_input
        else:
            print('Using default [rescale to 0-1] preprocess...')
            return lambda x: x / 255.0


    preprocess_input = get_preprocess_fn(model_type)
    preprocess_layer = tf.keras.layers.Lambda(lambda x: preprocess_input(tf.cast(x, tf.float32)), name='preprocessing')



    # Create datasets
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Extract class indices
    class_names = train_dataset.class_names
    class_indices = {name: i for i, name in enumerate(class_names)}
    print("\nClass Indices:", class_indices)    

    train_dataset = train_dataset.map(lambda x, y: (preprocess_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: (preprocess_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Pipeline optimization
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    print("---------------------------------------------------")

    return train_dataset, val_dataset, class_indices


def color_mapping(df, class_column):

    MAIN_COLORS = {
        'Black': ['Black', 'Charcoal'],
        'White': ['White', 'Off White', 'Cream'],
        'Red': ['Red', 'Maroon', 'Burgundy', 'Rust'],
        'Green': ['Green', 'Olive', 'Khaki', 'Sea Green', 'Lime Green', 'Fluorescent Green'],
        'Pink': ['Pink', 'Magenta', 'Rose', 'Lavender'],
        'Purple': ['Purple', 'Mauve'],
        'Yellow': ['Yellow', 'Mustard'],
        'Orange': ['Orange', 'Peach', 'Copper'],
        'Grey': ['Grey', 'Grey Melange', 'Steel'],
        'Silver': ['Silver', 'Metallic'],
        'Gold': ['Gold', 'Bronze'],
        'Multi': ['Multi'],  
        'Brown': ['Brown', 'Beige', 'Tan', 'Coffee Brown', 'Mushroom Brown', 'Taupe', 'Skin', 'Nude'],
        'Blue': ['Blue', 'Navy Blue', 'Turquoise Blue', 'Teal'] 
        }


    def map_color(color):
    
        for main_color, variants in MAIN_COLORS.items():
            if color in variants:
                return main_color
        return color  # If not in any group, return the same original color

    df[class_column] = df[class_column].apply(map_color)

    return df


def save_training_results(history, output_dir, time_stamp=""):
    """
    Save training history plots and reports to the output directory.

    Args:
        history: The History object returned by model.fit().
        output_dir: Path to the folder where results will be saved.
        time_stamp: Optional string for versioning file names (e.g., time).
    """

    # # Convert tensor values ​​to floats
    # history.history = {k: [float(v) if hasattr(v, 'numpy') else v for v in vals] 
    #                   for k, vals in history.history.items()}

    # Create folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_path = os.path.join(output_dir, f'accuracy_plot_{time_stamp}.png')
    plt.savefig(acc_path)
    plt.close()

    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_path = os.path.join(output_dir, f'loss_plot_{time_stamp}.png')
    plt.savefig(loss_path)
    plt.close()

    # Save CSV of history
    history_df = pd.DataFrame(history.history)
    csv_path = os.path.join(output_dir, f'history_{time_stamp}.csv')
    history_df.to_csv(csv_path, index=False)

    # Save textual report
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    report_path = os.path.join(output_dir, f'report_{time_stamp}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Final Validation Accuracy: {final_acc:.4f}\n")
        f.write(f"Final Validation Loss: {final_loss:.4f}\n")
        f.write(f"Epochs Trained: {len(history.history['loss'])}\n")

    print(f"Results saved to: {output_dir}")