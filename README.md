# AutoEncoders for Movie Recommendations

This project implements a simple AutoEncoder using PyTorch to recommend movies based on user ratings. The model is trained on the MovieLens dataset, specifically the 100k and 1M datasets. The goal is to predict the ratings a user would give to movies they haven't rated yet.

## Dataset

The MovieLens datasets are used, which are publicly available and can be downloaded using the following commands:

- [MovieLens 100K](http://files.grouplens.org/datasets/movielens/ml-100k.zip)
- [MovieLens 1M](http://files.grouplens.org/datasets/movielens/ml-1m.zip)

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- NumPy
- Pandas
- PyTorch

## Importing Libraries

The following libraries are required:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
```

## Preparing the Dataset

The datasets are read into Pandas DataFrames and converted into NumPy arrays for processing:

```python
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')
```

## Data Preprocessing

Convert the data into arrays where users are rows and movies are columns:

```python
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
```

## Model Architecture

The AutoEncoder is defined using PyTorch. The model consists of a series of fully connected layers with Sigmoid activations. The architecture is as follows:

1. **Input Layer**: The input is a vector representing user ratings for movies.
2. **First Hidden Layer**: Fully connected layer with 20 neurons and Sigmoid activation.
3. **Second Hidden Layer**: Fully connected layer with 10 neurons and Sigmoid activation.
4. **Third Hidden Layer**: Fully connected layer with 20 neurons and Sigmoid activation.
5. **Output Layer**: Fully connected layer that reconstructs the input vector.

```python
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
```

### Explanation of Components

- **Activation Function**: We use the `Sigmoid` activation function. It squashes the input values to be between 0 and 1, which is useful for ensuring the outputs are within a specific range.
  
- **Loss Function**: The `Mean Squared Error (MSE) Loss` is used to measure the difference between the predicted ratings and the actual ratings. This loss function is suitable for regression problems.

- **Optimizer**: `RMSprop` is used as the optimizer. It is an adaptive learning rate method designed to deal with non-stationary objectives by adapting the learning rate for each parameter.

## Training the Model

Train the Stacked AutoEncoder (SAE) for 200 epochs. During training, the model learns to reconstruct the input ratings. The following code trains the model:

```python
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
```

### Explanation

- **Epochs**: The number of times the entire dataset is passed through the network. We use 200 epochs.
- **Training Loop**: For each user, we create an input vector and a target vector (both are the same in this case). If the user has rated any movies, we perform a forward pass to get the output, calculate the loss, and perform a backward pass to update the weights.
- **Loss Adjustment**: The loss is adjusted by a mean corrector to normalize it.

## Testing the Model

Evaluate the performance of the SAE on the test set. The following code tests the model:

```python
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss / s))
```

### Explanation

- **Testing Loop**: Similar to the training loop, but here we use the test set. We calculate the loss for each user and average it to get the final test loss.

## Results

After training, the model prints the loss at each epoch and the final test loss, which indicates the performance of the model.


## License

This project is licensed under the MIT License.


