import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import click
import shap

def pad_gather_reduce(model, x, method="mean"):
    """
    pad a value or tensor across all processes and gather

    params:
        - trainer: a trainer that carries an accelerator object
        - x: a number or torch tensor to reduce
        - method: "mean", "sum", "max", "min"

    return:
        - the average tensor after maskin out 0's
        - None if the gather resulted in an empty tensor
    """

    assert method in [
        "mean",
        "sum",
        "max",
        "min",
    ], "This function has limited capabilities [sum, mean, max, min]"
    assert type(x) is not None, "Cannot reduce a None type object"

    # wait for everyone to arrive here before gathering

    if type(x) is not torch.Tensor:
        x = torch.tensor([x])

    # verify that the tensor is on the proper device
    x = x.to(model.device)

    # pad across processes
    padded_x = model.accelerator.pad_across_processes(x, dim=0)

    # gather across all procesess
    gathered_x = model.accelerator.gather(padded_x)

    # mask out zeros
    masked_x = gathered_x[gathered_x != 0]

    # if the tensor is empty, warn and return None
    if len(masked_x) == 0:
        click.secho(
            f"The call to this method resulted in an empty tensor after masking out zeros. The gathered tensor was this: {gathered_x} and the original value passed was: {x}.",
            fg="red",
        )
        return None

    if method == "mean":
        return torch.mean(masked_x)
    elif method == "sum":
        return torch.sum(masked_x)
    elif method == "max":
        return torch.max(masked_x)
    elif method == "min":
        return torch.min(masked_x)

def train(decoder, model, train_loader, device):
    model.train()
    for train_dataset in train_loader:
        snp, face = train_dataset
        #snp1 = snp.type(torch.float).to(device)
        images = face.to(device)
        text = snp.to(device)
        
        loss = model(text,images)
        model.update()

    return loss/len(train_loader)

@torch.no_grad()
def test(decoder, model, loader, device):
    model.eval()
    for data in loader:
        snp, face = data
        #snp1 = snp.type(torch.float).to(device)
        images = face.to(device)
        text1 = snp.to(device)
        
        loss = model(text1,images)

        #image_embeds = model.sample(text1) # (512, 512) - exponential moving averaged image embeddings

    return loss/len(loader)


@torch.no_grad()
def out(decoder, model, loader, device):
    model.eval()
    i = 1
    for data in loader:
        snp, face = data
        #snp1 = snp.type(torch.float).to(device)
        image = face.to(device)
        text = snp.to(device)

        background = text[:100]
        test_images = text[100:103]

        e = shap.Explainer(model.sample(), background)
        shap_values = e.shap_values(test_images)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        # plot the feature attributions
        shap.image_plot(shap_numpy, -test_numpy)

       # torch.save(image_embeds, (os.path.join( '/home/jmq/diffusion/out/output2/', str(i)+'_Latent.pt')))
        i = i + 1
    return i


def run(decoder, clip, model, train_loader, test_loader, epochs, writer, device):
    train_losses, test_losses = [], []
    i = 0 
    out_dir = '/share/home/jiaomingqi/diffusion5/out/checkpoint/'
    for epoch in range(0, epochs):
        #writer1 = SummaryWriter('/home/jmq/diffusion/out/tenbo1')
        #if epoch % 100 == 0:
            #embeds = out(decoder, model, test_loader, device)

        if epoch % 10 == 0:
            test_loss = test(decoder, model, test_loader, device)

            #writer1.add_scalar('test_loss', test_loss, epoch)
            print(epoch, 'test_loss', test_loss)

            orig_sim, pred_sim, pred_img_sim = report_cosine_sims(decoder,clip, model, test_loader, device)
            print(epoch, 'orig_sim', orig_sim, 'pred_sim', pred_sim, 'pred_img_sim', pred_img_sim)
            #orig_sim1, pred_sim1, pred_img_sim1 = report_cosine_sims(decoder, clip, model, train_loader, device)
            #print(epoch, 'orig_sim1', orig_sim1, 'pred_sim1', pred_sim1, 'pred_img_sim1', pred_img_sim1)
            #print(image_embeds)
        train_loss = train(decoder, model, train_loader, device)
        #writer1.add_scalar('train_loss', train_loss, epoch)
        #orig_sim, pred_sim, pred_img_sim = report_cosine_sims(clip, model, test_loader, device)
        #print(epoch, 'orig_sim', orig_sim, 'pred_sim', pred_sim, 'pred_img_sim', pred_img_sim)
        if epoch % 10 == 0:
            model.save(out_dir + str(epoch) + '_checkpoint')
        
        

        #print(i)
        print(epoch, 'train_loss', train_loss)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def report_cosine_sims(decoder, clip, model, loader, device):
    model.eval()
    for data in loader:
        snp, face = data
        #snp1 = snp.type(torch.float).to(device)
        images = face.to(device)
        text = snp.to(device)
        test_image_embeddings = clip.embed_image(images)
        text_embed = clip.embed_text(text)
        predicted_image_embeddings = model.sample(text)

        # prepare the text embedding
        text_embed = text_embed/ text_embed.norm(dim=1, keepdim=True)

        # prepare image embeddings
        test_image_embeddings = test_image_embeddings / test_image_embeddings.norm(
            dim=1, keepdim=True
        )

        predicted_image_embeddings = (
            predicted_image_embeddings
            / predicted_image_embeddings.norm(dim=1, keepdim=True)
        )

        # calculate similarities
        orig_sim = pad_gather_reduce(
           model, cos(text_embed, test_image_embeddings), method="mean"
        )

        pred_sim = pad_gather_reduce(
            model, cos(text_embed, predicted_image_embeddings), method="mean"
        )
        
        pred_img_sim = pad_gather_reduce(
            model,
            cos(test_image_embeddings, predicted_image_embeddings),
            method="mean",
        )
    return orig_sim, pred_sim, pred_img_sim
        
        