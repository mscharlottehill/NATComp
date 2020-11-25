from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        ## initialize tensor for inputs, and outputs
        x = torch.from_numpy(train_feats)
        y = torch.from_numpy(train_labels)
        N, D_in, H, D_out = 132, 6, 6, 1
        ## random init weights
        w1 = torch.randn(D_in, H)
        w2 = torch.randn(H, D_out)
        # take in 6 features (x/y/sinx/siny/x2/y2) for 8 hidden nodes
        self.hidden = nn.Linear(6, 8)
        # output 1 answer (label) from 8 nodes
        self.output = nn.Linear(8, 1)

        # Sigmoid activation/softmax output (basic NN)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def set_weights(self, name):
        with torch.no_grad():
            if name = "layer1":
                self.hidden.weight = torch.from_numpy([call PSO](initial weights))
            else:
                self.output.weight = torch.from_numpy([call PSO](hidden layer weights))

    # forward prop
    def forward(self, x):
        set_weights(layer1)
        x = F.relu(self.hidden(x))
        set_weights(layer2)
        x = F.relu(self.output(x))
        return x
