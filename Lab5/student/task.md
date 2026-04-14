This is an elective lab. You can solve it in a .py or .ipynb file.

On this lab, you're *not* allowed to use chatbots nor code agents. You *are* allowed to use the internet and search engines, including the google's built-in AI summary, but you're not allowed to ask it follow-up questions. At any moment, the person overseeing the laboratory can ask you to show and explain your code or any open tab. A failure to adhere to the rules awards -10 points for this lab.

Whenever you're asked to document the results, make a **screenshot of the entire screen** containing the relevant output. At the end of laboratories, send the code and screenshots (but not the saved models nor the data) via email.

# Random convolutional filters

Convolutional layers are very powerful feature extractors, even if they haven't been trained. Today, the goal is to implement a very shallow convolutional neural network and see if the convolutional part can be useful even if it stays random.

# Basic level (4p.)

1. (3p.) Create an extremely small convolutional neural network within the following constraints:
- only one convolutional layer
- an activation function of your choice
- an average global pooling layer
- a single linear layer
- at most 4000 trainable parameters in total

Here's a code that calculates the number of trainable parameters:
```
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Load the FashionMnist dataset divided into training and validation with the following code:
```
from torchvision import datasets, transforms
from torch.utils.data import Subset

transform = transforms.Compose([
    transforms.ToTensor(),
    # FashionMNIST channel statistics
    transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
])

train_dataset = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transform
)
train_dataset = Subset(train_dataset, range(0, len(train_dataset), 10))
val_dataset = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=transform
)
val_dataset = Subset(val_dataset, range(0, len(val_dataset), 10))
```

Use the training dataset to train your network (all of it, including the convolutional layer). Assume the dataset is balanced. Staying within the design constraints, improve the network and the choice of the hyperparameters (batch size, learning rate, momentum) until you reach 65% accuracy on the validation dataset for three consecutive epochs. Document the output.

2. (1p.) Change the code, so the convolutional layer is no longer trained (hint: optimizer creation). The only part of the network being trained should be the linear layer. Rerun the training code **from the beginning** and let it run for 15 epochs. Document the output, observing what will likely be a worse accuracy. Make sure the parameters of the convolutional layer didn't change.


# Intermediate level (4p.)

1. (1p.) Because the convolutional layer is now fixed, we can call it *a set of random convolutional filters*. While average global pooling was good for cnns, it works rather poorly with random convolutional filters. Replace it with you implementation of a special kind of pooling: the **proportion of positive values**.

PPV(X) = (Σ_{x in X} 1 if x>0 else 0) / (len(X))

To do this, create a new module that inherits from nn.Module or a function that inherits from torch.autograd.Function and implements the forward method. Don't implement the backward pass; we're not training the conv layer below anyway. Rerun the training code for 15 epochs and document the results.

2. (1p.) Our model is training poorly. Must be the bad conditioning. Modify PPV(X), keeping the essence of what it does, but changing it so the next layer trains better. 

3. (2p.) Our model is underfitting. We would like to make it more complex by increasing the number of features. Instead of changing the convolutional layer to have more filters (which would work, but would make the whole thing slower and eat VRAM), let's do something cheaper: change PPV to compare not to one bias=0, but to multiple predefined biases. Keep the changes you made to improve conditioning.

PPV(X) = [(Σ_{x in X} 1 if x>b else 0) / (len(X)) for b in ppv_biases]

Set ppv_biases to a fixed vector of 7 values of your choice. Modify the number of linear layer's inputs to account for the changes. Modify the hyperparameters to achieve 65% accuracy in three consecutive epochs on the validation data. Document the output.
Save the network with
```
torch.save(model.state_dict(), 'models/unpruned')
```
we will need it in the next level.

# Mastery level (4p.)

1. (2p.) Only a small part of our convolutional layer is actually useful. Let's prune these which weights in the linear layer are the weakest. Load the network with
```
model.load_state_dict(torch.load('models/unpruned', weights_only=True))
```
In a loop, turn to zero all weights leading from a given convolutional filter in the linear layer. Start from the filter with the lowest sum of absolute values of weights in the linear layer. Stop removing weights when the accuracy drops below 90% of the starting accuracy on the validation set. Document the outputs. Save the network with
```
torch.save(model.state_dict(), 'models/zero_weights')
```

2. (2p.) Load the network with
```
model.load_state_dict(torch.load('models/zero_weights', weights_only=True))
```
Create a smaller network that is a copy of that network, but without the pruned filters. The new networks' convolutional layer and linear layer should be smaller than the original (missing the pruned filters). Train the network for 10 epochs with a lower learning rate.
