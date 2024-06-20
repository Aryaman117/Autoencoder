# Autoencoder
Recommender System based on Stacked AutoEncoder model | Deep Learning, Neural Network.

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

The AutoEncoder is defined using PyTorch:

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

## Training the Model

Train the Stacked AutoEncoder (SAE) for 200 epochs:

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

## Testing the Model

Evaluate the performance of the SAE on the test set:

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

## Results

After training, the model prints the loss at each epoch and the final test loss, which indicates the performance of the model.

## License

This project is licensed under the MIT License.

