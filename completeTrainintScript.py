import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import json
import time
from datetime import datetime

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
                    
                    # 임시로 모두 0으로 설정 (추후 실제 레이블로 수정 필요)
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

class GlaucomaNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GlaucomaNet, self).__init__()
        # ResNet50을 기반으로 한 모델 생성
        self.model = resnet50(pretrained=True)
        # 마지막 fully connected layer를 새로운 분류기로 교체
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda'):
    """모델 학습 함수"""
    
    # 결과 저장을 위한 딕셔너리
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # 학습 시작 시간 기록
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        
        val_loss = running_loss / total
        val_acc = running_corrects.double() / total
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # 최고 성능 모델 저장
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved new best model with validation accuracy: {val_acc:.4f}')
        
        # Learning rate 조정
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch complete in {epoch_time:.0f}s')
    
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {history["best_val_acc"]:.4f} at epoch {history["best_epoch"]}')
    
    return history

def plot_training_history(history, save_path='training_history.png'):
    """학습 히스토리 시각화"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로드
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_refuge_dataset('.')
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    print("Initializing model...")
    model = GlaucomaNet()
    
    # Loss function과 optimizer 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # 모델 학습
    print("Starting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        NUM_EPOCHS,
        device
    )
    
    # 학습 결과 시각화
    plot_training_history(history)
    
    # 결과 저장
    results = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'best_val_acc': float(history['best_val_acc']),
        'best_epoch': history['best_epoch']
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()