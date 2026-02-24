# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task.
The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output.
It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="954" height="633" alt="image" src="https://github.com/user-attachments/assets/69eca247-4a7f-49b7-8cf7-3c1d21a57b76" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: S.Navadeep
### Register Number: 212224230180
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ----------------------------
# Student Details Function
# ----------------------------
def Navadeep():
    print("Name: S.Navadeep")
    print("Register Number: 212224230180")

# ----------------------------
# Load Dataset
# ----------------------------
dataset1 = pd.read_csv('MyMLData.csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

# ----------------------------
# Scaling
# ----------------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

# ----------------------------
# Neural Network Model
# ----------------------------
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

        Navadeep()
        print("Neural Network Regression Model Initialized")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create Model
ai_brain = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

# ----------------------------
# Training Function
# ----------------------------
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    Navadeep()

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

# Train Model
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

# ----------------------------
# Testing
# ----------------------------
with torch.no_grad():
    Navadeep()
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

# ----------------------------
# Plot Loss
# ----------------------------
loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

# ----------------------------
# Prediction
# ----------------------------
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)

scaled_input = scaler.transform(X_n1_1)
prediction = ai_brain(
    torch.tensor(scaled_input, dtype=torch.float32)
).item()

Navadeep()
print(f'Prediction for input 9: {prediction}')

```
## Dataset Information

<img width="142" height="372" alt="image" src="https://github.com/user-attachments/assets/8d3956e2-f315-4013-a02a-afc5825803e5" />

## OUTPUT  

<img width="498" height="394" alt="image" src="https://github.com/user-attachments/assets/6dfef1f5-5702-4de7-9cda-bd4ba6eeeaee" />

### Training Loss Vs Iteration Plot

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/c4a7a98d-54af-4448-ab87-341e1f3e924b" />

### New Sample Data Prediction

<img width="304" height="79" alt="image" src="https://github.com/user-attachments/assets/1c3e28b5-a0be-4668-9a6c-052679048038" />

## RESULT

Successfully executed the code to develop a neural network regression model.
