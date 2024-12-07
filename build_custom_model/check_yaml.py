import yaml
from pathlib import Path

def verify_dataset_setup():
    """Verify the dataset structure and data.yaml configuration"""
    
    print("🔍 Starting Dataset Verification...")
    
    # 1. Check data.yaml
    try:
        with open('data/data.yaml', 'r') as f:
            yaml_data = yaml.safe_load(f)
            print("\n📄 data.yaml contents:")
            for key, value in yaml_data.items():
                print(f"  {key}: {value}")
            
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in yaml_data:
                print(f"❌ Error: Missing '{key}' in data.yaml")
                return False
    except Exception as e:
        print(f"❌ Error reading data.yaml: {str(e)}")
        return False

    # 2. Check directory structure
    directories = {
        'train_images': Path('data/images/train'),
        'val_images': Path('data/images/val'),
        'train_labels': Path('data/labels/train'),
        'val_labels': Path('data/labels/val')
    }
    
    print("\n📁 Directory Structure:")
    for name, path in directories.items():
        if path.exists():
            files = len(list(path.glob('*.*')))
            print(f"  ✅ {name}: {path} ({files} files)")
        else:
            print(f"  ❌ {name}: {path} (directory not found)")
            
    # 3. Count files and verify matches
    train_images = len(list(directories['train_images'].glob('*.[jp][pn][g]')))
    train_labels = len(list(directories['train_labels'].glob('*.txt')))
    val_images = len(list(directories['val_images'].glob('*.[jp][pn][g]')))
    val_labels = len(list(directories['val_labels'].glob('*.txt')))
    
    print("\n📊 File Counts:")
    print(f"  Training Set:")
    print(f"    Images: {train_images}")
    print(f"    Labels: {train_labels}")
    print(f"  Validation Set:")
    print(f"    Images: {val_images}")
    print(f"    Labels: {val_labels}")
    
    # 4. Verify label format
    print("\n📋 Verifying label format...")
    label_errors = []
    
    def check_labels(label_dir):
        for label_file in Path(label_dir).glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            label_errors.append(f"Invalid format in {label_file.name}, line {line_num}")
                        elif not all(part.replace('.', '').replace('-', '').isdigit() for part in parts[1:]):
                            label_errors.append(f"Invalid numbers in {label_file.name}, line {line_num}")
                        elif not (0 <= int(parts[0]) < yaml_data['nc']):
                            label_errors.append(f"Invalid class ID in {label_file.name}, line {line_num}")
            except Exception as e:
                label_errors.append(f"Error reading {label_file.name}: {str(e)}")
    
    check_labels(directories['train_labels'])
    check_labels(directories['val_labels'])
    
    if label_errors:
        print("❌ Found label errors:")
        for error in label_errors:
            print(f"  - {error}")
    else:
        print("✅ All labels format verified")
    
    # 5. Summary
    print("\n📝 Summary:")
    if train_images == train_labels and val_images == val_labels:
        print("✅ Number of images matches number of labels")
    else:
        print("❌ Mismatch between number of images and labels")
    
    total_images = train_images + val_images
    print(f"\nTotal dataset size: {total_images} images")
    print(f"Split ratio: {train_images/total_images:.1%} train, {val_images/total_images:.1%} validation")

if __name__ == "__main__":
    verify_dataset_setup()