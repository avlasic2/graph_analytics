import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class EmbedClass(nn.Module):
    def __init__(
            self
            , dimensionEmbed:int
            , numberElements:int
            , sourceIndices:np.ndarray
            , targetIndices:np.ndarray
            , weights:np.ndarray
        ):
        super().__init__()
        self.embedVectors = nn.Parameter(
            nn.init.xavier_uniform_( torch.empty( numberElements, dimensionEmbed, requires_grad=True, dtype = torch.float64  )  )
        )
        self.matrixOperator = nn.Parameter(
            nn.init.kaiming_uniform_( torch.empty( dimensionEmbed,  dimensionEmbed, requires_grad=True, dtype = torch.float64  ) , mode='fan_out' )
        )
        self.biases = nn.init.zeros_( torch.empty( len(sourceIndices) ) )
        self._weights = torch.from_numpy(weights).type(torch.float64)
        self._sourceIndices = torch.from_numpy(sourceIndices).type(torch.long)
        self._targetIndices = torch.from_numpy(targetIndices).type(torch.long)

    def _loss(
            self
            , source:np.ndarray
            , target:np.ndarray
            , weight:np.ndarray
            , bias:np.ndarray
            , matrixOperator:np.ndarray
        )->np.ndarray:
        return torch.mean( torch.pow( torch.diag( source.mm( matrixOperator.mm( target.t() ) ) , 0)  - weight - bias, 2) )

    def _train(
            self
            , num_epoch:int
            , batch_size:int = 512
            , learning_rate:float = 0.05
        ):
        dataSet = SourceTargetDataSet(
              sourceIndices = self._sourceIndices
            , targetIndices = self._targetIndices
            , embeddings = self.embedVectors
            , weights = self._weights
            , bias = self.biases
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mtrxOprtr = self.matrixOperator.to(device)

        dataloader = DataLoader(dataSet, batch_size)
        optimizer = optim.Adam( self.parameters(), lr = learning_rate)

        for epoch in range(num_epoch):
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f'Epoch {epoch + 1}')
                tepoch.set_postfix( Train_loss = 0.0)
                for batch in tepoch:
                    source, target, weight, bias = batch
                    source, target, weight, bias = source.to(device), target.to(device), weight.to(device), bias.to(device)

                    optimizer.zero_grad()
                    lossAll = self._loss(source, target, weight, bias, mtrxOprtr)
                    ttlLoss = lossAll.item()
                    lossAll.backward()
                    optimizer.step()
                    tepoch.set_postfix( Train_loss = ttlLoss )


class SourceTargetDataSet(Dataset):
    def __init__(
            self
            , sourceIndices:np.ndarray
            , targetIndices:np.ndarray
            , embeddings:np.ndarray
            , weights:np.ndarray
            , bias:np.ndarray
        ):
        self._sourceIndices = sourceIndices
        self._targetIndices = targetIndices
        self._embeddings = embeddings
        self._weights = weights
        self._bias = bias
        self._sourceIndices = sourceIndices
        self._targetIndices = targetIndices

    def __getitem__(self, idx:int):
        return self._embeddings[ self._sourceIndices[idx] ] \
            , self._embeddings[ self._targetIndices[idx]] \
            , self._weights[idx] \
            , self._bias[idx]

    def __len__(self):
        return len(self._sourceIndices)