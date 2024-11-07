from pickle import TRUE
from re import X
from sre_constants import FAILURE
from unittest import loader
import torch
import torch.backends.cudnn as cudnn
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True    
torch.backends.cudnn.benchmark = True
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from glob import glob
from tqdm import tqdm
from torch.nn.functional import normalize as norm
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from spiralconv import SpiralConv

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out

class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out

class FACE_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform):
        super(FACE_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        
        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        '''
        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()
        '''
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def forward(self, x, *indices):
        
        z = self.encoder(x)
        
        return z
'''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.embedding_layer = nn.Embedding(4, 4)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            #nn.AvgPool1d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3,  padding=1),
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            #nn.AvgPool1d(kernel_size=2, stride=2)
            )
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=4 ,kernel_size=1, stride=1)
        self.fc = nn.Linear(4*7842, 16)


    def forward(self, x):

        embedded_matrix = self.embedding_layer(x)
        x_mean = torch.sum(embedded_matrix, 3)
        x = x_mean.permute(0, 2, 1)
        out = self.layer1(x)
        out = self.layer3(out)
        y = self.conv3(x)
        out += y
        out = F.relu(out)
        out = torch.flatten(out,1)
        out = self.fc(out)

        return out
'''

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.fc1 = nn.Linear(7842, 1024)
        self.embedding_layer = nn.Embedding(1,4)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.layer1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512, dropout=0.1, layer_norm_eps=1e-05)
        self.layer2 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512, dropout=0.1, layer_norm_eps=1e-05)
        #self.layer3 = nn.TransformerEncoderLayer(d_model=128, nhead=16, dim_feedforward=512, dropout=0.1, layer_norm_eps=1e-05)
        self.fc2 = nn.Linear(64*1024, 128)
        
    def forward(self, x):
        x = x.float()
        x = self.relu1(self.fc1(x))
        x = x.unsqueeze(2)  # 假设 x 的原始形状是 [32, 16], 这会改变 x 的形状为 [32, 16, 1]
        x = x.expand(-1, -1, 64)  # 现在你可以扩展第三个维度到 128, 结果形状为 [32, 16, 128]

        x = x.permute(1, 0, 2)   # 结果形状为[64, 32, 128]

        x1 = self.layer1(x) 
        x1 = x1 + x
        #x2 = self.layer2(x1)
        
        #x3 = self.layer2(x2)
        #x3 = x3 + x2

        x1 = x1.permute(1, 0, 2)
        x1 = torch.flatten(x1, start_dim = 1)
        out = self.relu2(self.fc2(x1))

        return out


class CLIP(nn.Module):
    
    def __init__(self, image_encoder, text_encoder, dim_text = 128,
        dim_image = 128,
        dim_latent = 128):
        super().__init__()

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent
        self.encoder1 = image_encoder
        self.encoder2 = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([1]))
        # text latent projection
        self.to_text_latent = nn.Linear(dim_text, dim_latent)
        # image latent projection
        self.to_image_latent = nn.Linear(dim_image, dim_latent)

    
    def forward(self, image, text):

        image_embeds = self.encoder1(image)
        image_features = self.to_image_latent(image_embeds)

        text_embeds = self.encoder2(text)
        text_features = self.to_text_latent(text_embeds)

        #text_latents, image_latents = map(l2norm, (text_features, image_features))
        # normalized features
        #image_features = image_features / image_features.norm(keepdim=True)
        #text_features = text_features / text_features.norm(keepdim=True)

        
        text_latents, image_latents = map(l2norm, (text_features, image_features))
        return  text_latents
    
    def embed_text(self,text):
        text_embeds = self.encoder2(text)
        text_features = self.to_text_latent(text_embeds)
        #print(text_features)
        return text_features

    def embed_image(self,image):
        image_embeds = self.encoder1(image)
        image_features = self.to_image_latent(image_embeds)
        #print(image_features)
        return image_features