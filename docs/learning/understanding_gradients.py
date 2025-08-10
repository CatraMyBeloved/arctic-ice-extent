import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# PART 1: WHAT IS A GRADIENT? (CALCULUS REFRESHER)
# =============================================================================
print("="*60)
print("PART 1: GRADIENTS - THE INTUITION")
print("="*60)

print("\n1. Single Variable: Derivative = Slope")
print("If f(x) = x², then f'(x) = 2x")
print("At x=3: slope = 6 (function increasing quickly)")
print("At x=0: slope = 0 (function at minimum)")
print("At x=-2: slope = -4 (function decreasing)")

# Visualize this
x = np.linspace(-4, 4, 100)
y = x**2
dy_dx = 2*x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
ax1.plot([3], [9], 'ro', markersize=8, label='Point (3, 9)')
ax1.plot([3-0.5, 3+0.5], [9-3, 9+3], 'r--', linewidth=2, label='Slope = 6')
ax1.set_title('Function f(x) = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True)

ax2.plot(x, dy_dx, 'g-', linewidth=2, label="f'(x) = 2x")
ax2.plot([3], [6], 'ro', markersize=8, label='Slope at x=3')
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_title('Derivative = Slope at each point')
ax2.set_xlabel('x')
ax2.set_ylabel("f'(x)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\n2. Multiple Variables: Gradient = Vector of Partial Derivatives")
print("If f(x,y) = x² + y², then:")
print("∂f/∂x = 2x  (how much f changes when x changes)")
print("∂f/∂y = 2y  (how much f changes when y changes)")
print("Gradient = [∂f/∂x, ∂f/∂y] = [2x, 2y]")
print("The gradient points in the direction of steepest increase!")

# =============================================================================
# PART 2: COMPUTATIONAL GRAPHS
# =============================================================================
print("\n" + "="*60)
print("PART 2: HOW PYTORCH TRACKS OPERATIONS")
print("="*60)

print("\n1. Simple example: z = (x + y)²")
print("Let's see how PyTorch builds the computational graph:")

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

print(f"x = {x}")
print(f"y = {y}")

# Each operation creates a node in the computational graph
a = x + y       # a = 5
print(f"a = x + y = {a}")
print(f"a.grad_fn: {a.grad_fn}")  # Shows how 'a' was computed

z = a**2        # z = 25
print(f"z = a² = {z}")
print(f"z.grad_fn: {z.grad_fn}")  # Shows how 'z' was computed

print("\nComputational graph:")
print("x(2) \\")
print("      + → a(5) → **2 → z(25)")
print("y(3) /")

print("\n2. Computing gradients with chain rule:")
print("We want: ∂z/∂x and ∂z/∂y")
print("Chain rule: ∂z/∂x = (∂z/∂a) × (∂a/∂x)")

z.backward()  # This computes all gradients automatically!

print(f"\n∂z/∂x = {x.grad}")  # Should be 10
print(f"∂z/∂y = {y.grad}")    # Should be 10

print("\nManual calculation:")
print("z = (x + y)²")
print("∂z/∂x = 2(x + y) × 1 = 2(2 + 3) = 10 ✓")
print("∂z/∂y = 2(x + y) × 1 = 2(2 + 3) = 10 ✓")

# =============================================================================
# PART 3: BACKPROPAGATION STEP BY STEP
# =============================================================================
print("\n" + "="*60)
print("PART 3: BACKPROPAGATION STEP BY STEP")
print("="*60)

print("\n1. More complex example: Neural network computation")
print("Let's trace through a mini neural network:")

# Reset gradients
x = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

print(f"Input: x = {x}")
print(f"Weights: w1 = {w1}, w2 = {w2}")
print(f"Bias: b = {b}")

# Forward pass
h = w1 * x + b      # h = 2*1 + (-1) = 1
h_relu = torch.relu(h)  # h_relu = max(0, 1) = 1
output = w2 * h_relu    # output = 0.5 * 1 = 0.5

print(f"\nForward pass:")
print(f"h = w1*x + b = {h}")
print(f"h_relu = ReLU(h) = {h_relu}")
print(f"output = w2*h_relu = {output}")

# Suppose target is 2.0, so loss = (output - target)²
target = torch.tensor(2.0)
loss = (output - target)**2  # loss = (0.5 - 2)² = 2.25

print(f"\nLoss = (output - target)² = {loss}")

print("\n2. Backward pass (automatic!):")
loss.backward()

print(f"∂loss/∂w1 = {w1.grad}")
print(f"∂loss/∂w2 = {w2.grad}")
print(f"∂loss/∂b = {b.grad}")
print(f"∂loss/∂x = {x.grad}")

print("\n3. Manual verification (chain rule):")
print("∂loss/∂output = 2(output - target) = 2(0.5 - 2) = -3")
print("∂output/∂w2 = h_relu = 1")
print("∂loss/∂w2 = (∂loss/∂output) × (∂output/∂w2) = -3 × 1 = -3 ✓")

print("\n∂output/∂h_relu = w2 = 0.5")
print("∂h_relu/∂h = 1 (since h > 0, ReLU derivative = 1)")
print("∂h/∂w1 = x = 1")
print("∂loss/∂w1 = -3 × 0.5 × 1 × 1 = -1.5 ✓")

# =============================================================================
# PART 4: WHY GRADIENTS MATTER FOR LEARNING
# =============================================================================
print("\n" + "="*60)
print("PART 4: GRADIENT DESCENT - LEARNING FROM GRADIENTS")
print("="*60)

print("\n1. Gradient tells us how to improve!")
print("If ∂loss/∂w is negative → increasing w will decrease loss")
print("If ∂loss/∂w is positive → decreasing w will decrease loss")
print("Rule: w_new = w_old - learning_rate × ∂loss/∂w")

print(f"\nCurrent: w1 = {w1.data}, loss = {loss}")
print(f"Gradient: ∂loss/∂w1 = {w1.grad}")
print("Since gradient is negative, we should increase w1")

learning_rate = 0.1
new_w1 = w1.data - learning_rate * w1.grad
print(f"Updated: w1 = {new_w1}")

print("\n2. Visualization: Loss landscape")
# Create a simple loss landscape
w_values = np.linspace(-1, 3, 100)
losses = []

for w_val in w_values:
    # Simulate the same computation with different w1 values
    h_sim = w_val * 1.0 + (-1.0)  # w*x + b
    h_relu_sim = max(0, h_sim)
    output_sim = 0.5 * h_relu_sim
    loss_sim = (output_sim - 2.0)**2
    losses.append(loss_sim)

plt.figure(figsize=(10, 6))
plt.plot(w_values, losses, 'b-', linewidth=2, label='Loss landscape')
plt.plot([w1.data], [loss.data], 'ro', markersize=10, label=f'Current position (w1={w1.data})')
plt.arrow(w1.data, loss.data, -w1.grad*0.3, 0, head_width=0.1, head_length=0.05,
          fc='red', ec='red', label='Gradient direction')
plt.xlabel('w1 value')
plt.ylabel('Loss')
plt.title('Loss Landscape - Gradient Points Toward Lower Loss')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# PART 5: AUTOMATIC DIFFERENTIATION IN ACTION
# =============================================================================
print("\n" + "="*60)
print("PART 5: PYTORCH'S AUTOMATIC DIFFERENTIATION")
print("="*60)

print("\n1. Complex functions - no manual chain rule needed!")

def complex_function(x):
    """A complicated function to differentiate"""
    return torch.sin(x**2) * torch.exp(-x) + torch.log(x + 1)

x = torch.tensor(2.0, requires_grad=True)
y = complex_function(x)

print(f"f(x) = sin(x²)×e^(-x) + ln(x+1)")
print(f"f(2) = {y}")

y.backward()
print(f"f'(2) = {x.grad}")

# Compare with numerical approximation
h = 1e-6
numerical_grad = (complex_function(torch.tensor(2.0 + h)) -
                  complex_function(torch.tensor(2.0 - h))) / (2 * h)
print(f"Numerical approximation: {numerical_grad}")
print(f"Automatic differentiation is exact!")

print("\n2. Multiple outputs (Jacobian)")
print("When function has multiple outputs, PyTorch can compute all gradients:")

x = torch.tensor([1.0, 2.0], requires_grad=True)

# Function with 2 inputs, 2 outputs
y1 = x[0]**2 + x[1]**3  # y1 = x₁² + x₂³
y2 = x[0] * x[1]        # y2 = x₁ × x₂

print(f"Input: x = {x}")
print(f"y1 = x₁² + x₂³ = {y1}")
print(f"y2 = x₁ × x₂ = {y2}")

# To get gradients for y1
y1.backward(retain_graph=True)  # retain_graph=True to compute more gradients
print(f"∇y1 = [∂y1/∂x₁, ∂y1/∂x₂] = {x.grad}")

x.grad.zero_()  # Clear gradients
y2.backward()
print(f"∇y2 = [∂y2/∂x₁, ∂y2/∂x₂] = {x.grad}")

print("\n" + "="*60)
print("SUMMARY: WHY THIS MATTERS FOR NEURAL NETWORKS")
print("="*60)
print("1. Neural networks are just complex functions with many parameters")
print("2. We want to minimize loss by adjusting parameters")
print("3. Gradients tell us which direction to move each parameter")
print("4. PyTorch automatically computes all gradients via backpropagation")
print("5. This enables us to train networks with millions of parameters!")
print("\n🎯 Next: Understanding how this applies to LSTMs and time series")