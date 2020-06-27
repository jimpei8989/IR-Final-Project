import torch
from torch import nn


class BertModel(nn.Module):
    def __init__(self, bertModel, clfHiddenDim=1024):
        super().__init__()
        self.bert = bertModel

        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, clfHiddenDim),
            nn.ReLU(),

            nn.Linear(clfHiddenDim, 1),
            nn.Sigmoid()
        )

        self.freezeBert()
        
    def forward(self, documentTokens, queryTokens):
        documentEmbedding = self.bert(documentTokens)[0][:, 0]
        queryEmbedding = self.bert(queryTokens)[0][:, 0]

        return self.classifier(torch.cat([documentEmbedding, queryEmbedding], dim=1))

    def freezeBert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreezeBert(self, last = 2):
        for layer in list(self.bert.encoder.layer.childrens())[-last:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        