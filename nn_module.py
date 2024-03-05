# model class 
import time
from importlib import reload

import numpy as np
from torch import nn 
# from crfseg import CRF
from tqdm.auto import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F


def tqdm_if(obj, condition):
    return tqdm(obj) if condition else obj


class CRF(nn.Module):
    """
    Class for learning and inference in conditional random field model using mean field approximation
    and convolutional approximation in pairwise potentials term.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions of input tensors.
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    n_iter : int
        Number of iterations in mean field approximation.
    requires_grad : bool
        Whether or not to train CRF's parameters.
    returns : str
        Can be 'logits', 'proba', 'log-proba'.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    smoothness_theta : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel.
        If it is a sequence its length must be equal to ``n_spatial_dims``.
    """

    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5, requires_grad=True,
                 returns='logits', smoothness_weight=1, smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns
        self.requires_grad = requires_grad

        self._set_param('smoothness_weight', smoothness_weight)
        self._set_param('inv_smoothness_theta', 1 / np.broadcast_to(smoothness_theta, n_spatial_dims))

    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, spatial_spacings=None, display_tqdm=False):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. the CNN's output.
        spatial_spacings : array of floats or None
            Array of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.
            None is equivalent to all ones. Used to adapt spatial gaussian filters to different inputs' resolutions.
        display_tqdm : bool
            Whether to display the iterations using tqdm-bar.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``
            with logits or (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        assert len(spatial) == self.n_spatial_dims

        # binary segmentation case
        if n_classes == 1:
            x = torch.cat([x, torch.zeros(x.shape).to(x)], dim=1)

        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        negative_unary = x.clone()

        for i in tqdm_if(range(self.n_iter), display_tqdm):
            # normalizing
            x = F.softmax(x, dim=1)

            # message passing
            x = self.smoothness_weight * self._smoothing_filter(x, spatial_spacings)

            # compatibility transform
            x = self._compatibility_transform(x)

            # adding unary potentials
            x = negative_unary - x

        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = F.softmax(x, dim=1)
        elif self.returns == 'log-proba':
            output = F.log_softmax(x, dim=1)
        else:
            raise ValueError("Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")

        if n_classes == 1:
            output = output[:, 0] - output[:, 1] if self.returns == 'logits' else output[:, 0]
            output.unsqueeze_(1)

        return output

    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        return torch.stack([self._gaussian_filter(x[i], self.inv_smoothness_theta, spatial_spacings[i], self.filter_size)
                            for i in range(x.shape[0])])

    @staticmethod
    def _gaussian_filter(x, inv_theta, spatial_spacing, filter_size):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        inv_theta : torch.tensor
            Tensor of shape ``(len(spatial),)``
        spatial_spacing : sequence of len(spatial) floats
        filter_size : sequence of len(spatial) ints

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        """
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filtering
            kernel = CRF.create_gaussian_kernel1d(inv_theta[i], spatial_spacing[i], filter_size[i]).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel, padding=(filter_size[i] // 2,))

            # reshape back to (n, *spatial)
            x = x.squeeze(1).view(*shape_before_flatten, x.shape[-1]).transpose(-1, dim)

        return x

    @staticmethod
    def create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        """
        Parameters
        ----------
        inverse_theta : torch.tensor
            Tensor of shape ``(,)``
        spacing : float
        filter_size : int

        Returns
        -------
        kernel : torch.tensor
            Tensor of shape ``(filter_size,)``.
        """
        distances = spacing * torch.arange(-(filter_size // 2), filter_size // 2 + 1).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, n_classes, *spatial)``.

        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels, labels.unsqueeze(1)).to(x)
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Input tensors must be broadcastable.

        Parameters
        ----------
        label1 : torch.Tensor
        label2 : torch.Tensor

        Returns
        -------
        compatibility : torch.Tensor
        """
        return -(label1 == label2).float()



class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_size, nclasses, device='cpu') -> None:
        super().__init__()

        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)
        self.lstm_model = nn.LSTM(embedding_dim, hidden_size//2, bidirectional=True).to(device)
        self.ffwd_lay = nn.Linear(hidden_size, nclasses).to(device)
        self.crf = CRF(n_spatial_dims=1, returns='proba')

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.criterion = nn.CrossEntropyLoss()
        
        self.loss_history = []

    def forward(self, batch):   
        out, _ = self.lstm_model(batch)
        out = self.ffwd_lay(out)
        out = self.crf.forward(out.T.unsqueeze(0))
        return out[0].T
    

    def fit(self, train_X, train_Y, valid_X, valid_Y, nepochs, lr, device):
        self.train()
        self.to(device)

        for g in self.optim.param_groups:
            g['lr'] = lr
        
        for ep in tqdm(range(nepochs)):
            eploss = 0
        
            if ep % 20 == 0:
                for g in self.optim.param_groups:
                    g['lr'] = lr/2  

            for i, (batch_X, batch_Y) in tqdm(enumerate(zip(train_X, train_Y))):
                self.zero_grad()
                
                predict = self.forward(batch_X.to(device))
                loss = self.criterion(predict, batch_Y.to(device))
                loss.backward()
                self.optim.step()

                eploss += loss.item()
                self.loss_history.append(loss.item())

            printbool = ep % (nepochs//10) == 0 if nepochs > 10 else True
            if printbool:
                with torch.no_grad():
                    TP, TN, FP, FN = 0., 0., 0., 0.
                    for batch_X, batch_Y in tqdm(zip(valid_X, valid_Y)):
                        predict = torch.argmax(self.forward(batch_X.to(device)), dim=1).cpu()
                        TP += ((predict == batch_Y) & (predict != 0)).sum()
                        TN += ((predict == batch_Y) & (predict == 0)).sum()
                        FP += ((predict != batch_Y) & (predict != 0)).sum()
                        FN += ((predict != batch_Y) & (predict == 0)).sum()
                    p_metric_valid = TP / (TP + FP)
                    r_metric_valid = TP / (TP + FN)

                    print(f'Loss: {eploss/len(train_X):.3f}, Valid precision: {p_metric_valid:.3f}, Valid recall: {r_metric_valid:.3f}')
                    
 