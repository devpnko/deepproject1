import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_accuracies):
    """
    학습 히스토리 시각화
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Accuracy plot
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # 데이터 로드
    train_dataset, val_dataset, test_dataset = load_dataset('path/to/data')
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE
    )
    
    # 모델 초기화
    model = GlaucomaNet()
    
    # Loss function과 optimizer 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # 모델 학습
    train_losses, val_accuracies = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS
    )
    
    # 학습 히스토리 시각화
    plot_training_history(train_losses, val_accuracies)
    
    # 테스트 세트에서 평가
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()