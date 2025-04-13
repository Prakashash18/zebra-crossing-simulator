import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os


class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PoseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def train_model():
    # Load data
    X = np.load('data/features/X.npy')
    y = np.load('data/features/y.npy')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create datasets and dataloaders
    train_dataset = PoseDataset(X_train, y_train)
    test_dataset = PoseDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # Initialize model
    input_size = X.shape[1]
    num_classes = len(np.unique(y))
    model = PoseClassifier(input_size, num_classes)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {train_loss/len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.2f}%')
    
    # Save model and scaler
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/pose_classifier.pth')
    np.save('models/scaler.npy', {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    train_model() 