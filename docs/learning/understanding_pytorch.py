import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("PYTORCH SYNTAX GUIDE - BUILDING NETWORKS STEP BY STEP")
print("=" * 70)

print("\n🎯 THE PYTORCH WORKFLOW:")
print("1. Create class inheriting from nn.Module")
print("2. Add layers and activations in __init__")
print("3. Define forward pass in forward() method")
print("4. Create training loop with loss and backprop")

# =============================================================================
# STEP 1: CREATE THE NETWORK CLASS
# =============================================================================
print("\n" + "=" * 50)
print("STEP 1: CREATE THE NETWORK CLASS")
print("=" * 50)


class MyFirstNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Always call parent constructor first
        super(MyFirstNetwork, self).__init__()

        # Define layers (these will have trainable parameters)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Define activation functions (these don't have parameters)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Alternative: you can also use functional activations (see forward method)

    def forward(self, x):
        # Define how data flows through the network
        x = self.input_layer(x)  # Linear transformation
        x = self.relu(x)  # Non-linearity
        x = self.hidden_layer(x)  # Another linear transformation
        x = self.relu(x)  # Another non-linearity
        x = self.output_layer(x)  # Final layer
        x = self.sigmoid(x)  # Final activation
        return x


print("✅ SYNTAX BREAKDOWN:")
print("• class MyNetwork(nn.Module): - inherit from PyTorch base class")
print("• super().__init__() - call parent constructor")
print("• self.layer_name = nn.LayerType() - define layers")
print("• def forward(self, x): - define data flow")
print("• return x - output the result")

# =============================================================================
# STEP 2: AVAILABLE LAYERS AND ACTIVATIONS
# =============================================================================
print("\n" + "=" * 50)
print("STEP 2: AVAILABLE LAYERS AND ACTIVATIONS")
print("=" * 50)

print("\n🧱 COMMON LAYERS:")
print("• nn.Linear(in_features, out_features) - fully connected layer")
print("• nn.Conv2d(in_channels, out_channels, kernel_size) - 2D convolution")
print("• nn.LSTM(input_size, hidden_size) - LSTM layer (for sequences!)")
print("• nn.Dropout(p=0.5) - randomly zero some neurons during training")
print("• nn.BatchNorm1d(num_features) - normalize layer inputs")

print("\n⚡ COMMON ACTIVATIONS:")
print("• nn.ReLU() - max(0, x)")
print("• nn.Sigmoid() - 1/(1 + e^(-x))")
print("• nn.Tanh() - tanh(x)")
print("• nn.LeakyReLU() - leaky version of ReLU")
print("• nn.Softmax(dim=1) - for probability distributions")


# Example of different ways to define activations
class ActivationExamples(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

        # Method 1: Define as layers
        self.relu_layer = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)

        # Method 1: Use predefined layer
        x = self.relu_layer(x)

        # Method 2: Use functional form (same result)
        # x = F.relu(x)

        return x


print("\n💡 TWO WAYS TO USE ACTIVATIONS:")
print("• As layers: self.relu = nn.ReLU(), then x = self.relu(x)")
print("• As functions: x = F.relu(x) (import torch.nn.functional as F)")

# =============================================================================
# STEP 3: BUILDING NETWORKS WITH DIFFERENT PATTERNS
# =============================================================================
print("\n" + "=" * 50)
print("STEP 3: DIFFERENT NETWORK ARCHITECTURES")
print("=" * 50)

print("\n🏗️ PATTERN 1: SEQUENTIAL (simplest)")
# When you just stack layers one after another
simple_net = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

print("Sequential syntax:")
print("net = nn.Sequential(layer1, activation1, layer2, activation2, ...)")
print("Pro: Very simple")
print("Con: Can't do complex connections")

print("\n🏗️ PATTERN 2: CUSTOM CLASS (more flexible)")


class FlexibleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super().__init__()

        # Define all layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

        # Define other components
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # You can do anything here!
        x = self.layer1(x)
        x = self.batch_norm(x)  # Normalize
        x = F.relu(x)  # Activate
        x = self.dropout(x)  # Regularize

        # Skip connection (advanced)
        residual = x
        x = self.layer2(x)
        x = F.relu(x)
        x = x + residual  # Add skip connection

        x = self.layer3(x)  # Final layer
        return x


print("Custom class syntax:")
print("• Define layers in __init__")
print("• Define data flow in forward()")
print("• Can do complex operations, skip connections, etc.")

# =============================================================================
# STEP 4: CREATING AND USING THE NETWORK
# =============================================================================
print("\n" + "=" * 50)
print("STEP 4: CREATING AND USING THE NETWORK")
print("=" * 50)

# Create the network
net = MyFirstNetwork(input_size=3, hidden_size=10, output_size=1)

print(f"✅ Network created: {type(net)}")
print(f"Parameters: {sum(p.numel() for p in net.parameters())} trainable weights")

# Check network structure
print(f"\n📋 Network structure:")
print(net)

# Test with dummy data
dummy_input = torch.randn(5, 3)  # 5 samples, 3 features each
output = net(dummy_input)

print(f"\n🧪 Test run:")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# =============================================================================
# STEP 5: SETTING UP TRAINING COMPONENTS
# =============================================================================
print("\n" + "=" * 50)
print("STEP 5: SETTING UP TRAINING COMPONENTS")
print("=" * 50)

print("\n🎯 LOSS FUNCTIONS:")
print("• nn.MSELoss() - for regression (predicting numbers)")
print("• nn.CrossEntropyLoss() - for classification")
print("• nn.L1Loss() - mean absolute error")
print("• nn.BCELoss() - binary classification")

print("\n🚀 OPTIMIZERS:")
print("• optim.SGD(model.parameters(), lr=0.01) - basic gradient descent")
print("• optim.Adam(model.parameters(), lr=0.001) - adaptive learning rate")
print("• optim.RMSprop() - another adaptive optimizer")

# Create training components
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print(f"\n✅ Training components:")
print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")

# =============================================================================
# STEP 6: THE TRAINING LOOP SYNTAX
# =============================================================================
print("\n" + "=" * 50)
print("STEP 6: THE TRAINING LOOP SYNTAX")
print("=" * 50)

# Create dummy training data
n_samples = 100
X_train = torch.randn(n_samples, 3)
y_train = torch.randn(n_samples, 1)

print("🔁 TRAINING LOOP TEMPLATE:")
training_code = '''
for epoch in range(num_epochs):
    # FORWARD PASS
    predictions = model(inputs)
    loss = criterion(predictions, targets)

    # BACKWARD PASS
    optimizer.zero_grad()    # Clear old gradients
    loss.backward()          # Compute new gradients
    optimizer.step()         # Update weights

    # Optional: print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
'''
print(training_code)

print("\n🏃 ACTUAL TRAINING:")
num_epochs = 300
losses = []

for epoch in range(num_epochs):
    # Forward pass
    predictions = net(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record loss
    losses.append(loss.item())

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print(f"Final loss: {losses[-1]:.6f}")

# =============================================================================
# STEP 7: EVALUATION AND TESTING
# =============================================================================
print("\n" + "=" * 50)
print("STEP 7: EVALUATION AND TESTING")
print("=" * 50)

print("\n🧪 EVALUATION SYNTAX:")
evaluation_code = '''
# Switch to evaluation mode
model.eval()

# Don't track gradients during evaluation
with torch.no_grad():
    test_predictions = model(test_inputs)
    test_loss = criterion(test_predictions, test_targets)
'''
print(evaluation_code)

# Example evaluation
net.eval()  # Switch to evaluation mode
with torch.no_grad():
    test_input = torch.randn(10, 3)
    test_predictions = net(test_input)

print(f"✅ Evaluation completed")
print(f"Test predictions shape: {test_predictions.shape}")

# Visualize training progress
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_train.numpy(), predictions.detach().numpy(), alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# STEP 8: COMPLETE EXAMPLE TEMPLATE
# =============================================================================
print("\n" + "=" * 50)
print("STEP 8: COMPLETE PYTORCH TEMPLATE")
print("=" * 50)

complete_template = '''
import torch
import torch.nn as nn
import torch.optim as optim

# 1. DEFINE NETWORK
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. CREATE NETWORK AND TRAINING COMPONENTS
model = MyNetwork(input_size=10, hidden_size=20, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. TRAINING LOOP
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. EVALUATION
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
'''

print(complete_template)

print("\n" + "=" * 70)
print("🎯 PYTORCH SYNTAX SUMMARY")
print("=" * 70)
print("✅ Class: class MyNet(nn.Module)")
print("✅ Constructor: super().__init__() + define layers")
print("✅ Forward: def forward(self, x) + return result")
print("✅ Loss: criterion = nn.MSELoss()")
print("✅ Optimizer: optimizer = optim.Adam(model.parameters())")
print("✅ Training: forward → loss → zero_grad → backward → step")
print("✅ Evaluation: model.eval() + torch.no_grad()")

print("\n🚀 READY FOR TIME SERIES!")
print("Next: LSTM layers for sequential data like ice extent!")