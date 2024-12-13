import os
import shutil
from sklearn.model_selection import train_test_split
import random
import glob
import yaml

def create_dataset_splits(
    train_ratio=0.8,     # 80% for training, 20% for validation
    random_seed=42       # For reproducibility
):
    # Move to the root directory
    file_classes = 'data/labels/classes.txt'
    shutil.move('data/labels/train/classes.txt', file_classes)

    # Define source directories
    source_images = "data/images/train"
    source_labels = "data/labels/train"
    
    # Create directory structure
    dirs = {
        'train_images': "data/images/train",
        'train_labels': "data/labels/train",
        'val_images': "data/images/val",
        'val_labels': "data/labels/val"
    }
    
    # Create validation directories if they don't exist
    os.makedirs(dirs['val_images'], exist_ok=True)
    os.makedirs(dirs['val_labels'], exist_ok=True)
    print(f"Created validation directories")

    # Get all image files
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(source_images, f'*{ext}')))
    
    # Split the dataset
    train_images, val_images = train_test_split(
        image_files,
        train_size=train_ratio,
        random_state=random_seed
    )
    
    # Counter for moved files
    moved_count = {'train': len(train_images), 'val': 0}
    
    # Move validation files
    for img_path in val_images:
        # Move image
        img_filename = os.path.basename(img_path)
        shutil.move(
            img_path,
            os.path.join(dirs['val_images'], img_filename)
        )
        
        # Move corresponding label
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_src = os.path.join(source_labels, label_filename)
        if os.path.exists(label_src):
            shutil.move(
                label_src,
                os.path.join(dirs['val_labels'], label_filename)
            )
            moved_count['val'] += 1
    
    # Print summary
    print("\nDataset Split Summary:")
    print(f"Training set: {moved_count['train']} images and labels")
    print(f"Validation set: {moved_count['val']} images and labels")
    
    # Create data.yaml file
    
    names = [n for n in open(file_classes).read().split('\n') if n != '']
    yaml_content = {
        'train': './data/images/train',  # Relative paths
        'val': './data/images/val',
        'nc': len(names),  # number of classes
        'names': names  # class names
    }
    
    yaml_path = 'data/data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"\nCreated data.yaml at: {yaml_path}")
    
    return dirs, yaml_path

if __name__ == "__main__":
    # Run the split
    dirs, yaml_path = create_dataset_splits(
        train_ratio=0.8  # 80% training, 20% validation
    )