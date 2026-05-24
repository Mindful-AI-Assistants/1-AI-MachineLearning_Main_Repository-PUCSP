import torch.nn as nn


class MLP_EMNIST(nn.Module):
    def __init__(self, n_classes: int = 47):
        super().__init__()
        self.linear_relu_stack = nn.Sequential( # Sequência de camadas lineares e ReLU
            nn.Flatten(), # Transforma a imagem 28x28 em um vetor de 784 elementos
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Adiciona dropout para evitar overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Adiciona dropout para evitar overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        '''Função de propagação para frente do modelo'''
        return self.linear_relu_stack(x)