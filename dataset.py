import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json

class GlaucomaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_refuge_dataset(base_dir, image_size=(224, 224)):
    """REFUGE 데이터셋 로드"""
    print("Loading REFUGE dataset...")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터 경로 설정
    train_dir = os.path.join(base_dir, 'REFUGE', 'train', 'Images')
    val_dir = os.path.join(base_dir, 'REFUGE', 'val', 'Images')
    test_dir = os.path.join(base_dir, 'REFUGE', 'test', 'Images')
    
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Test directory: {test_dir}")
    
    # 레이블 정보 로드
    def load_labels(split):
        json_path = os.path.join(base_dir, 'REFUGE', split, 'index.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return None

    # 데이터 로드 함수
    def load_images_from_dir(directory, label_data=None):
        images = []
        labels = []
        image_paths = []
        
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return [], [], []
            
        for img_name in sorted(os.listdir(directory)):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(directory, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    image_paths.append(img_name)
                    
                    # 레이블 할당 (임시로 모두 0으로 설정)
                    # 실제로는 label_data에서 읽어와야 함
                    labels.append(0)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return images, labels, image_paths
    
    # 각 세트 로드
    train_images, train_labels, train_paths = load_images_from_dir(train_dir)
    val_images, val_labels, val_paths = load_images_from_dir(val_dir)
    test_images, test_labels, test_paths = load_images_from_dir(test_dir)
    
    print(f"Found {len(train_images)} training images")
    print(f"Found {len(val_images)} validation images")
    print(f"Found {len(test_images)} test images")
    
    # Dataset 객체 생성
    train_dataset = GlaucomaDataset(train_images, train_labels, transform=train_transform)
    val_dataset = GlaucomaDataset(val_images, val_labels, transform=transform)
    test_dataset = GlaucomaDataset(test_images, test_labels, transform=transform)
    
    return train_dataset, val_dataset, test_dataset

def main():
    # 현재 작업 디렉토리 출력
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    try:
        # 데이터셋 로드
        train_dataset, val_dataset, test_dataset = load_refuge_dataset('.')
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=0  # Mac에서의 문제를 방지하기 위해 0으로 설정
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0
        )
        
        # 데이터 샘플 확인
        print("\nChecking data batch:")
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()