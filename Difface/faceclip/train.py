import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import einsum
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def get_last_shared_layer(model):
        return model.get_last_shared_layer()

def run(model, decoder, train_loader, test_loader, epochs, optimizer, scheduler, writer, device):
    train_losses, test_losses = [], []

    for epoch in range(0, epochs):
        if epoch % 5 == 0:   
            test_loss_clip, test_loss_encoder, cos = test(model, decoder, test_loader, device)
            print('test_loss_clip', test_loss_clip, 'test_loss_encoder', test_loss_encoder, 'cos', cos)

        train_loss_clip, train_loss_encoder, cos1= train(model, decoder, optimizer, train_loader, device)
        scheduler.step()
        #writer1.add_scalar('train_loss', train_loss, epoch)
        print(epoch,  'train_loss_clip', train_loss_clip, 'train_loss_encoder', train_loss_encoder, 'cos1', cos1) # 'test_loss_i',  test_loss_i,'test_loss_t', test_loss_t )
        #if epoch % 100 == 0:
            #text_features , image_features = out(model, decoder, train_loader, device)
        if epoch % 10 == 0:
            writer.save_checkpoint(model, decoder, optimizer, scheduler, epoch)



def train(model, decoder, optimizer, loader, device):

    error1 = []
    error2 = []
    model.train()
    decoder.train()
    total_loss1 = 0
    total_loss2 = 0
    total_loss_i = 0
    total_loss_t = 0
    for data in loader:

        
        snp, face = data
        #snp1 = snp.type(torch.float).to(device)
        #print(face.shape)
        image1 = face.to(device)
        text1 = snp.to(device)
        #image1=image1.float()
        #text1=text1.float()
        optimizer.zero_grad()
        text_features , image_features  = model(image1, text1)
        #print(text_features.shape)
        image_embeds = model.embed_image(image1)
        #text_embeds = model.embed_text(text1)
        
        out = decoder(image_embeds)
        #print(text_features , image_features)
        a = cos(image_features, text_features)
        #print(a)
        error1.append(a)

        logit_scale = model.logit_scale.exp()
        logits1 = logit_scale * image_features @ text_features.t()
        logits2 = logit_scale * text_features @ image_features.t()
        labels = torch.arange(logits1.size(0)).to(device)
        cross = nn.CrossEntropyLoss()
        loss_i = cross(logits1, labels)
        loss_t = cross(logits2, labels)
        #print(loss_i)
        #print(loss_t)
        loss1 = (loss_i+loss_t)/2
        #print(loss1)
        loss2 = F.l1_loss(out, image1, reduction='mean')
        loss = 1*loss1 + 10*loss2
        loss.backward()

        #theta1 = grads['z'].view(-1)
        optimizer.step()
        total_loss1 += loss1.cpu().detach().numpy().item()
        total_loss2 += loss2.cpu().detach().numpy().item()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        del image1, text1, text_features , image_features, loss1, loss2
        torch.cuda.empty_cache()
    new_errors1 = torch.cat(error1, dim=0)  # [n_total_graphs, num_nodes]
    mean_error1 = new_errors1.view((-1, )).mean()
    #new_errors2 = torch.cat(error2, dim=0)  # [n_total_graphs, num_nodes]
    #mean_error2 = new_errors2.view((-1, )).mean()
    return total_loss1 / len(loader), total_loss2 / len(loader), mean_error1#, mean_error2

@torch.no_grad()
def test(model, decoder, loader, device):
    errors = []
    model.eval()
    decoder.eval()
    total_loss1 = 0
    total_loss2 = 0
    total_loss_i = 0
    total_loss_t = 0
    for data in loader:

        snp, face = data
        #snp1 = snp.type(torch.float).to(device)
        #print(len(snp1))
        image1 = face.to(device)
        text1 = snp.to(device)
       # image1=image1.float()
        #text1=text1.float()
        text_features , image_features  = model(image1, text1)
        #print(text_features , image_features)
        image_embeds = model.embed_image(image1)
        #text_embeds = model.embed_text(text1)
        out = decoder(image_embeds)

        a = cos(image_features, text_features)
        errors.append(a)
        
        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits1 = logit_scale * image_features @ text_features.t()
        logits2 = logit_scale * text_features @ image_features.t()

        labels = torch.arange(logits1.size(0)).to(device)

        cross = nn.CrossEntropyLoss()
        loss_i = cross(logits1, labels)
        loss_t = cross(logits2, labels)
        
        loss1 = (loss_i+loss_t)/2
        
        loss2 = F.l1_loss(out, image1, reduction='mean')
 
        # coefficient1 = nn.Parameter(torch.Tensor([1])).cuda()
        total_loss1 += loss1.cpu().detach().numpy().item()
        total_loss2 += loss2.cpu().detach().numpy().item()
        del image1, text1, text_features , image_features, loss1, loss2
        torch.cuda.empty_cache()
    new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]
    mean_error = new_errors.view((-1, )).mean()
    return total_loss1 / len(loader), total_loss2 / len(loader), mean_error
