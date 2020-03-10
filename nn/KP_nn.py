import torch.nn as nn
import torch.nn.functional as F



class KP_NN(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.ll1 = nn.Linear(num_input,12000)
        self.ll2 = nn.Linear(12000, 12000)
        self.ll3 = nn.Linear(12000, 12000)
        # self.ll4 = nn.Linear(9000, 9000)
        # self.ll5 = nn.Linear(9000, 9000)
        # self.ll6 = nn.Linear(9000, 9000)
        # self.ll7 = nn.Linear(5000, 1000)
        # self.ll8 = nn.Linear(1000, 1000)
        self.ll9 = nn.Linear(12000, num_output)

    def forward(self, X):
        X = F.relu(self.ll1(X))
        X = F.relu(self.ll2(X))
        X = F.relu(self.ll3(X))
        # X = F.relu(self.ll4(X))
        # X = F.relu(self.ll5(X))
        # X = F.relu(self.ll6(X))
        # X = F.relu(self.ll7(X))
        # X = F.relu(self.ll8(X))
        X = self.ll9(X)
        # return F.log_softmax(X, dim=1)
        return F.sigmoid(X)