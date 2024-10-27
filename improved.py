import os
import json
import numpy as np
from collections import Counter

def verify_and_load_labels(base_dir):
    """레이블 정보 확인 및 로드"""
    print("\nVerifying label information...")
    
    # train set의 레이블 확인
    train_json = os.path.join(base_dir, 'REFUGE', 'train', 'index.json')
    val_json = os.path.join(base_dir, 'REFUGE', 'val', 'index.json')
    
    def load_json_labels(json_path):
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # index.json 파일의 구조에 따라 적절히 수정 필요
                labels = [item.get('label', None) for item in data]
                return labels
        return None
    
    train_labels = load_json_labels(train_json)
    val_labels = load_json_labels(val_json)
    
    if train_labels:
        print(f"Train set label distribution: {Counter(train_labels)}")
    if val_labels:
        print(f"Validation set label distribution: {Counter(val_labels)}")
    
    return train_labels, val_labels

def load_refuge_dataset_with_labels(base_dir, image_size=(224, 224)):
    """레이블 정보를 포함한 데이터셋 로드"""
    print("Loading REFUGE dataset with verified labels...")
    
    # 기존의 transform 코드는 유지
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
    
    # 레이블 정보 로드
    train_labels, val_labels = verify_and_load_labels(base_dir)
    
    # 이미지 로드 함수
    def load_images_from_dir(directory):
        images = []
        for img_name in sorted(os.listdir(directory)):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(directory, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return images
    
    # 데이터 로드
    train_dir = os.path.join(base_dir, 'REFUGE', 'train', 'Images')
    val_dir = os.path.join(base_dir, 'REFUGE', 'val', 'Images')
    
    train_images = load_images_from_dir(train_dir)
    val_images = load_images_from_dir(val_dir)
    
    # 레이블이 없는 경우 임시 레이블 생성 (검증용)
    if not train_labels:
        print("Warning: No label information found. Creating temporary labels for verification...")
        train_labels = [i % 2 for i in range(len(train_images))]  # 임시로 0과 1을 번갈아가며 할당
    if not val_labels:
        val_labels = [i % 2 for i in range(len(val_images))]
    
    print(f"\nDataset statistics:")
    print(f"Training: {len(train_images)} images")
    print(f"Training label distribution: {Counter(train_labels)}")
    print(f"Validation: {len(val_images)} images")
    print(f"Validation label distribution: {Counter(val_labels)}")
    
    # Dataset 생성
    train_dataset = GlaucomaDataset(train_images, train_labels, transform=train_transform)
    val_dataset = GlaucomaDataset(val_images, val_labels, transform=transform)
    
    return train_dataset, val_dataset

def calculate_metrics(outputs, labels):
    """세부적인 성능 지표 계산"""
    _, preds = torch.max(outputs, 1)
    accuracy = (preds == labels).float().mean()
    
    # 클래스별 정확도
    class_correct = torch.zeros(2)
    class_total = torch.zeros(2)
    for i in range(2):
        mask = (labels == i)
        if mask.sum() > 0:
            class_correct[i] = ((preds == labels) & mask).float().sum()
            class_total[i] = mask.float().sum()
    
    class_accuracies = class_correct / class_total
    
    return {
        'accuracy': accuracy.item(),
        'class_accuracies': class_accuracies.tolist(),
        'predictions': preds.cpu().numpy(),
        'true_labels': labels.cpu().numpy()
    }

def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, 
                           scheduler, num_epochs=50, device='cuda'):
    """향상된 메트릭을 포함한 모델 학습"""
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': [],
        'best_val_acc': 0.0, 'best_epoch': 0
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_metrics = []
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_metrics.append(calculate_metrics(outputs, labels))
        
        # Validation phase
        model.eval()
        val_metrics = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_metrics.append(calculate_metrics(outputs, labels))
        
        # 메트릭 평균 계산
        avg_train_metrics = average_metrics(train_metrics)
        avg_val_metrics = average_metrics(val_metrics)
        
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Train Accuracy: {avg_train_metrics["accuracy"]:.4f}')
        print(f'Class Accuracies: {avg_train_metrics["class_accuracies"]}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {avg_val_metrics["accuracy"]:.4f}')
        
        # 최고 성능 모델 저장
        if avg_val_metrics["accuracy"] > history['best_val_acc']:
            history['best_val_acc'] = avg_val_metrics["accuracy"]
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step(val_loss/len(val_loader))
    
    return history