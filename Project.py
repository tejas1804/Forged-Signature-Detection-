# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt
import os
import itertools

# Define the CNN Model (AlexNet)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.linear_layers(x)
        return x

# Define Evaluation Metrics
def confusion_metrics(conf_matrix):
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    conf_accuracy = (TP + TN) / (TP + TN + FP + FN)
    conf_sensitivity = TP / (TP + FN)
    conf_specificity = TN / (TN + FP)
    conf_precision = TN / (TN + FP)
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print(f"Accuracy(%): {conf_accuracy * 100:.2f}")
    print(f"Sensitivity(Recall): {conf_sensitivity:.2f}")
    print(f"Specificity: {conf_specificity:.2f}")
    print(f"Precision: {conf_precision:.2f}")
    print(f"F1 Score: {conf_f1:.2f}")

# Train the Model
def train_model(data_path, save_path, batch_size=32, epochs=40):
    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.RandomCrop(227),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    targets = dataset.targets
    X_train, X_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.2, shuffle=True, random_state=101)
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

    model = Net()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005, weight_decay=0.0005)
    
    loss_values = []
    accuracy_values = []
    
    for epoch in range(epochs):
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            output = model(imgs)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(output, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(X_train):.4f}, Accuracy: {epoch_accuracy:.2f}%")
        loss_values.append(running_loss / len(X_train))
        accuracy_values.append(epoch_accuracy)
        
    torch.save(model, os.path.join(save_path, 'alexnet_bs32_0.0005.pkl'))
    savetxt('loss_values.csv', np.array(loss_values), delimiter=';')
    savetxt('accuracy_values.csv', np.array(accuracy_values), delimiter=';')
    plt.plot(loss_values, label='Loss')
    plt.plot(accuracy_values, label='Accuracy')
    plt.legend()
    plt.savefig('Training_metrics.png')
    plt.show()
    
    return model

# Test the Model
def test_model(model, data_path):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model.eval()
    y_true, y_pred, correct, total = [], [], 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            output = model(imgs)
            _, preds = torch.max(output, 1)
            y_true.extend(labels)
            y_pred.extend(preds)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    confusion_metrics(cm)
    plt.matshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}", ha='center', va='center', color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion_matrix.png')
    plt.show()

# Main Execution
data_path = input("Enter dataset path: ")
save_path = input("Enter model save path: ")
model = train_model(data_path, save_path)
test_model(model, data_path)

