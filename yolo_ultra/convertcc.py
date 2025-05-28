import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import subprocess

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def convert_to_engine():
    COLOR_CLASSES = ['beige_brown', 'black', 'blue', 'gold', 'green', 'grey', 
                     'orange', 'pink', 'purple', 'red', 'white', 'yellow']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimpleCNN(len(COLOR_CLASSES))
    checkpoint = torch.load("ccmodel.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    torch.onnx.export(model, dummy_input, "ccmodel.onnx", export_params=True, 
                      opset_version=11, input_names=['input'], output_names=['output'])
    
    # Convert to TensorRT
    cmd = ["trtexec", "--onnx=ccmodel.onnx", "--saveEngine=ccmodel.engine", "--fp16"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup
    if os.path.exists("ccmodel.onnx"):
        os.remove("ccmodel.onnx")
    
    print("✅ ccmodel.engine created successfully!")

if __name__ == "__main__":
    convert_to_engine()