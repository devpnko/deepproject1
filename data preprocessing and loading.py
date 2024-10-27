import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def load_dataset(data_dir, image_size=(224, 224)):
    """
    데이터셋을 로드하고 전처리하는 함수
    """
    images = []
    labels = []
    
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터 증강을 위한 transform
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ORIGA 데이터셋 로드
    origa_dir = os.path.join(data_dir, 'ORIGA')
    for label in ['normal', 'glaucoma']:
        label_dir = os.path.join(origa_dir, label)
        label_idx = 0 if label == 'normal' else 1
        
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Train/Val/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Dataset 객체 생성
    train_dataset = GlaucomaDataset(X_train, y_train, transform=train_transform)
    val_dataset = GlaucomaDataset(X_val, y_val, transform=transform)
    test_dataset = GlaucomaDataset(X_test, y_test, transform=transform)
    
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    DataLoader 객체 생성
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader