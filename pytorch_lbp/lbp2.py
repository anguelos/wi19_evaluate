import numpy as np
import torch
from matplotlib import pyplot as plt
import math

class LBPdifferential(torch.nn.Module):
    @staticmethod
    def gaussian_filter(center, var=1, filter_size=5, filter_range=None):
        if filter_range is None:
            x = torch.linspace(center - var * 2, center + var * 2, filter_size)
        else:
            x = torch.linspace(filter_range[0], filter_range[1], filter_size) - center
        fx = (1 / (2 * math.pi * var) ** .5) * torch.exp(-(x ** 2 / (2 * var)))
        return fx

    @staticmethod
    def get_simple_LBP(nb_samples, radius, variance=1, filter_size=15):
        angles = torch.linspace(0, 2 * np.pi, nb_samples + 1)[:-1]
        filter_bank = torch.zeros([nb_samples, 1, filter_size, filter_size])
        filter_range = -(filter_size - 1) / 2, (filter_size - 1) / 2
        for n in range(nb_samples):
            filter_x = LBPdifferential.gaussian_filter(radius * np.cos(angles[n]), variance, filter_size, filter_range)
            filter_y = LBPdifferential.gaussian_filter(radius * np.sin(angles[n]), variance, filter_size, filter_range)
            filter = filter_x[None, :] * filter_y[:, None]
            filter_bank[n, 0, :, :] = filter / filter.sum()
        #W=torch.Tensor(filter_bank)
        W=filter_bank
        b = torch.zeros(nb_samples, dtype=torch.float)-.01
        offsets_x = torch.zeros(nb_samples,dtype=torch.int32)
        offsets_y = torch.zeros(nb_samples, dtype=torch.int32)
        return W, b, None, None


    def __init__(self,radius,nb_samples,variance=1):
        super(LBPdifferential,self).__init__()
        self.W,self.b,self.offsets_x,self.offsets_y=LBPdifferential.get_simple_LBP(nb_samples=nb_samples,radius=radius,filter_size=int(math.ceil(radius))*2+1, variance=variance)

    def forward(self,X):
        if offset_x is None and offset_y is None:
            res = torch.nn.functional.conv2d(X, self.W, groups=1) + b.view(1, -1, 1, 1)
        elif offset_x is not None and offset_y is not None:
            nb_samples=offset_x.size()[0]
            pad_x_range= (offset_x.max()/2)*2
            pad_y_range = (offset_y.max() / 2) * 2
            padders=[torch.nn.ZeroPad2d((offset_x[n],pad_x_range-offset_x[n],offset_y[n],pad_y_range-offset_y[n])) for n in range(nb_samples)]
            padded_outputs=[]
            for n in range(nb_samples):
                padded_input=padders[n](img)
                padded_outputs.append(torch.nn.functional.conv2d(img, W[n:n+1,n], groups=1))
            res=torch.cat(padded_outputs,dim=1)+ b.view(1, -1, 1, 1)
        else:
            raise Exception("")
        return res



class LBPAlphabetLayer(torch.nn.Module):
    @staticmethod
    #def create_alphabet_tensor():
    def __init__(self):
        super(LBPAlphabetLayer,self).__init__()







def apply_weights(img,W,b,offset_x=None, offset_y=None):
    if offset_x is None and offset_y is None:
        res=torch.nn.functional.conv2d(img, W, groups=1) + b.view(1, -1, 1, 1)
    elif offset_x is not None and offset_y is not None:
        nb_samples=offset_x.size()[0]
        pad_x_range= (offset_x.max()/2)*2
        pad_y_range = (offset_y.max() / 2) * 2
        padders=[torch.nn.ZeroPad2d((offset_x[n],pad_x_range-offset_x[n],offset_y[n],pad_y_range-offset_y[n])) for n in range(nb_samples)]
        padded_outputs=[]
        for n in range(nb_samples):
            padded_input=padders[n](img)
            padded_outputs.append(torch.nn.functional.conv2d(img, W[n:n+1,n], groups=1))
        res=torch.cat(padded_outputs,dim=1)+ b.view(1, -1, 1, 1)
    else:
        raise Exception("")
    return res




def plot_8_filters(activations,central_activation=None,save_fname=None):
    activations,central_activation=activations.view(activations.size()[1:]),central_activation.view(central_activation.size()[1:])
    fig,ax_list=plt.subplots(3,3)
    if central_activation is not None:
        ax_list[1][1].imshow(central_activation.detach().numpy()[0,:,:],cmap="gray")
    if activations.size()[0] > 0:
        ax_list[1][2].imshow(activations[0, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 1:
        ax_list[0][2].imshow(activations[1, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 2:
        ax_list[0][1].imshow(activations[2, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 3:
        ax_list[0][0].imshow(activations[3, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 4:
        ax_list[1][0].imshow(activations[4, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 5:
        ax_list[2][0].imshow(activations[5, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 6:
        ax_list[2][1].imshow(activations[6, :, :].detach().numpy(),cmap="gray")
    if activations.size()[0] > 7:
        im=ax_list[2][2].imshow(activations[7, :, :].detach().numpy(),cmap="gray")
    fig.colorbar(im,ax=ax_list.ravel().tolist())
    for n in range(3):
        for k in range(3):
            ax_list[n][k].get_xaxis().set_visible(False)
            ax_list[n][k].get_yaxis().set_visible(False)
    #fig.tight_layout()
    if save_fname is not None:
        fig.savefig(save_fname)
    return fig

if __name__=="__main__":
    filter_size=31
    variance=.3
    W,b,offset_x,offset_y=LBPdifferential.get_simple_LBP(8,radius=5, variance=variance,filter_size=filter_size,variance=.3)
    #plt.imshow(W.view(-1,filter_size).numpy())
    #plt.show()
    image = torch.zeros(1,1,200,300)
    image[0, 0, :, :] = torch.Tensor(np.abs(np.linspace(-2+0j,2+0j,200)[:,None]+np.linspace(0-2j,0+2j,300)[None,:]))
    activations = apply_weights(image, W, b)
    delta_activations=activations-image[:,:,15:-15,15:-15]
    fig=plot_8_filters(torch.sigmoid(torch.relu(10*delta_activations)),image)
    plt.show()
    #plt.imshow(torch.relu(delta_activations).sum(dim=1)[0,:,:].numpy())
    #plt.colorbar()
    #plt.show()

