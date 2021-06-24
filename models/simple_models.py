from torch import nn

# Just some model, nothing really exsciting
class SimpleNeuralNetwork(nn.Module):
    # TODO you could at least make sure that input dimensions is correct
    # or input is being croppped or whatever
    def __init__(self):
        # always init super
        super(SimpleNeuralNetwork, self).__init__()
        
        # can also store hyperparameters in model
        # self.hyperparameters = {"learning_rate": 1e-3, "batch_size": 64, "epochs": 5}
        
        # model
        self.linear_relu_stack = nn.Sequential(
            # for linear layers you might want to flatten images
            nn.Flatten(),
            # actual model
            nn.Linear(400*400, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.ReLU(),
            # softmax at the end
            nn.Softmax(dim=1)
        )

    # defines forward pass (never call yourself)
    def forward(self, x):
        pred_probab = self.linear_relu_stack(x)
        y_pred = pred_probab.argmax(1)
        return y_pred