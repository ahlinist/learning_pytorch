import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the Data from CSV File
data = pd.read_csv('dataset.csv')
aux_data = pd.read_csv('dataset_aux.csv')

# Normalize input (i1) and outputs (o1, o2)
t_mean, t_std = data['i1'].mean(), data['i1'].std()
o1_mean, o1_std = data['o1'].mean(), data['o1'].std()
o2_mean, o2_std = data['o2'].mean(), data['o2'].std()

# Normalize the input (i1)
t = torch.tensor(((data['i1'] - t_mean) / t_std).values, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

# Normalize the outputs (o1, o2)
coords = torch.tensor(
    np.stack([(data['o1'] - o1_mean) / o1_std, (data['o2'] - o2_mean) / o2_std], axis=1),
    dtype=torch.float32
)  # Shape: (N, 2)

# Define the same OrbitPredictor class (no changes here)
class OrbitPredictor(nn.Module):
    def __init__(self):
        super(OrbitPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),  # 1 input (time) -> 64 hidden units
            nn.ReLU(),
            nn.Linear(64, 64),  # Hidden layer
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: 2 (x, y)
        )

    def forward(self, t):
        return self.model(t)

class OrbitRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=2, num_layers=2):
        super(OrbitRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, t_sequence):
        # Ensure input has 3 dimensions: (batch_size, seq_len, input_size)
        if t_sequence.dim() == 2:  # If shape is (batch_size, input_size)
            t_sequence = t_sequence.unsqueeze(1)  # Add seq_len dimension -> (batch_size, 1, input_size)

        rnn_out, _ = self.rnn(t_sequence)  # Shape: (batch_size, seq_len, hidden_size)

        # Take the output of the last time step
        output = self.fc(rnn_out[:, -1, :])  # Shape: (batch_size, output_size)
        return output


# Instantiate the model
#model = OrbitPredictor()
model = OrbitRNN()

# Step 3: Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 4: Train the Model
epochs = 50
losses = []

for epoch in range(epochs):
    outputs = model(t)  # Forward pass
    loss = criterion(outputs, coords)  # Compute loss

    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    # Log loss
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# Plot the training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Step 5: Use the Trained Model for Inference
# Normalize the test input values
#t_test_raw = np.array([100000000000.0, 7400000000000.0, 8500000000000.0])
t_test_raw = np.linspace(0, 10000000000000, 20)
t_test = torch.tensor((t_test_raw - t_mean) / t_std, dtype=torch.float32).unsqueeze(1)  # Normalize test input

# Predict normalized outputs
predicted_coords_normalized = model(t_test).detach().numpy()

# Denormalize the predictions
predicted_coords_x = predicted_coords_normalized[:, 0] * o1_std + o1_mean  # Denormalize o1
predicted_coords_y = predicted_coords_normalized[:, 1] * o2_std + o2_mean  # Denormalize o2

# Visualize the results
plt.scatter(data['o1'], data['o2'], label="True Orbit (Ground Truth)", color="blue")
plt.scatter(aux_data['o1'], aux_data['o2'], label="Expected Orbit", color="green")
plt.scatter(predicted_coords_x, predicted_coords_y, label="Predicted Orbit", color="red")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True vs Predicted Orbit")
plt.grid()
plt.show()
