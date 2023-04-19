import torch
import torch.nn as nn


class SimilarClassifier(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.similarity_func = nn.CosineSimilarity()
        # self.sigmoid = nn.Sigmoid()

    def _forward_once(self, x):
        x = self.fc1(x)
        return self.relu(x)

    def forward(self, x1, x2):
        x1 = self._forward_once(x1)
        x2 = self._forward_once(x2)

        return self.similarity_func(x1, x2)


class SentimentClassifier(nn.Module):
    """Binary sentiment classifier"""
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    
class SentimentClassifier1(nn.Module):
    """Binary sentiment classifier - Similar with using sigmoid"""
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)[:, 0]  # return the probability of positive sentiment

class EntailmentClassifier(nn.Module):
    """Multi Class Classifier"""
    def __init__(self, hidden_size, num_classes=3, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size* 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _forward_once(self, x):
        x = self.fc1(x)
        return self.relu(x)

    def forward(self, x1, x2):
        x1 = self._forward_once(x1)
        x2 = self._forward_once(x2)

        # stack x1 and x2
        x = torch.cat((x1, x2), dim=1)
        x = self.fc2(x)
        return self.softmax(x)
    
    def inference(self, x1, x2):
        return self.forward(x1, x2)[:, 0]  # entailment score