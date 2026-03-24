# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
A Recurrent Neural Network (RNN) is a type of deep learning model designed to handle sequential data, such as time series like stock prices. It processes previous inputs through loops, allowing it to capture temporal dependencies and patterns over time. When used for stock price prediction, the RNN analyzes historical price data to learn trends and make future price estimates. Its ability to remember information across sequences makes it suitable for modeling the dynamic and seasonal nature of stock markets. Overall, RNNs help improve forecast accuracy by leveraging past data to inform future predictions.



## DESIGN STEPS
### STEP 1: 
Load and normalize data, create sequences.
### STEP 2: 
Covert data to tensors and set up DataLoader.


### STEP 3: 
Define the RNN model architecture.


### STEP 4: 
Summarize, compile with loss and optimizer.


### STEP 5: 
Train the model with loss tracking.


### STEP 6: 
Predict on test data, polt actual vs. predicted prices.




## PROGRAM

### Name: D Karthikeyan

### Register Number: 212224230115

```python
# Define RNN Model
class RNNModel(nn.Module):
    # write your code here
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
      super(RNNModel, self).__init__()
      self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc  = nn.Linear(hidden_size,output_size)
      
    def forward(self,x):
      out,_ = self.rnn(x)
      out = self.fc(out[:,-1,:])
      return out 




# Train the Model
def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch = x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f'Epoch [{epoch+1} / {epochs}], Loss: {total_loss / len(train_loader):.4f}')

  print('Name:    D Karthikeyan')
  print('Register Number:  212224230115')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
train_model(model,train_loader,criterion,optimizer)

```

### OUTPUT

## Training Loss Over Epochs Plot


<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/4151913c-cb48-4837-bd11-e859471e759a" />

## True Stock Price, Predicted Stock Price vs time
<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/b7ab82d7-6eec-4cea-ae97-30ca67fa1117" />


### Predictions
Include the predictions on test data

## RESULT
<img width="883" height="266" alt="image" src="https://github.com/user-attachments/assets/7904c7d6-a4b1-4864-ad9b-89911640e71a" />
