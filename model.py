import torch.nn as nn

class LogReg(nn.Module):
    """
    Simple Logistic Regression for linear evaluation of frozen SSL features.
    """
    def __init__(self, dim: int, n_class: int):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(dim, n_class)

        # Standard initialization for linear probes
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)