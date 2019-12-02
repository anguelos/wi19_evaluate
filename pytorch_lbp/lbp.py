import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2
#import skimage.filters as filters
import math
import torchvision
import matplotlib

def gaussian_filter(center, var=1, filter_size=5):
    x = torch.linspace(center-var*2,center+var*2,filter_size)
    fx = (1 / (2 * math.pi * var) ** .5) * torch.exp(-(x ** 2 / (2 * var)))
    return fx

class OffsetRadialFeatures(torch.nn.Module):
    def __init__(self,input_channels, nb_samples, radius):
        super(OffsetRadialFeatures, self).__init__()
        self.filter_bank, self.left, self.top = \
            OffsetRadialFeatures.create_offset_lbp_filter_bank(input_channels,radius,nb_samples)
        self.nb_samples=nb_samples
        self.radius=radius
        self.right=self.left.max()-self.left
        self.bottom=self.top.max()-self.top
        self.filter_bank=torch.nn.Parameter(self.filter_bank)
        self.zero_padders=[torch.nn.ZeroPad2d((self.left[n],self.right[n],self.top[n],self.bottom[n])) for n in range(nb_samples+1)]
        self.zero_padders=torch.nn.ModuleList(self.zero_padders)

    def forward(self,x):
        padded_convolved=[]
        #print repr(x)
        central_padded=self.zero_padders[self.nb_samples](x)
        #print self.filter_bank.size()
        #print self.filter_bank[self.nb_samples:self.nb_samples+1,:,:,:].size()
        central_padded=torch.nn.functional.conv2d(central_padded, weight=self.filter_bank[self.nb_samples:self.nb_samples+1,:,:,:], groups=1)
        for n in range(self.nb_samples):
            padded=self.zero_padders[n](x)
            peripheral=torch.nn.functional.conv2d(padded, weight=self.filter_bank[n:n+1], groups=1)
            padded_convolved.append(peripheral-central_padded)
        convolved=torch.cat(padded_convolved,dim=1)
        return torch.relu(convolved), central_padded
        #return convolved, central_padded


    @staticmethod
    def create_offset_lbp_filter_bank(input_channels,radius,nb_samples):
        filter_zero_y = filter_zero_x = np.ceil(radius)
        angles=np.linspace(0,np.pi*2,nb_samples+1)[:-1]
        filter_bank=[]
        offsets_y=[]
        offsets_x=[]
        for n in range(len(angles)):
            angle=angles[n]+.001
            x=np.cos(angle)*radius
            y=np.sin(angle)*radius
            if x == np.round(x):
                x_left=int(x)
                x_right=x_left
                x_left_coef=1.0
                x_right_coef=0.0
            else:
                x_left=int(np.floor(x))
                x_right=x_left+1
                x_right_coef=x-x_left
                x_left_coef=1-x_right_coef

            if y == np.round(y):
                y_top=int(y)
                y_bottom=y_top
                y_top_coef=1.0
                y_bottom_coef=0.0
            else:
                y_top=int(np.floor(y))
                y_bottom = y_top+1
                y_bottom_coef=y-y_top
                y_top_coef=1-y_bottom_coef

            coef_sum = x_left_coef * y_top_coef + x_left_coef * y_bottom_coef + x_right_coef*y_top_coef + x_right_coef * y_bottom_coef

            offsets_y.append(filter_zero_y + y_top)
            offsets_x.append(filter_zero_x + x_left)
            y_bottom-=y_top
            x_right-=x_left
            y_top=0
            x_left=0
            filter = [[0,0],[0,0]]
            filter[y_top][x_left]+= x_left_coef*y_top_coef/coef_sum
            filter[y_bottom][x_left]+= x_left_coef * y_bottom_coef/coef_sum
            filter[y_top][x_right]+=x_right_coef*y_top_coef/coef_sum
            filter[y_bottom][x_right]+= x_right_coef * y_bottom_coef/coef_sum
            filter_bank.append(filter)
        #appending central image
        filter_bank.append([[1,0],[0,0]])
        offsets_y.append(filter_zero_y)
        offsets_x.append(filter_zero_x)

        filter_bank=torch.Tensor(filter_bank).view(nb_samples+1,1,2,2)
        filter_bank=filter_bank.repeat(1,input_channels,1,1)
        return torch.Tensor(filter_bank),torch.IntTensor(offsets_x),torch.IntTensor(offsets_y)


def create_lbp_filter_bank(radius,nb_samples,filter_size=None):
    if filter_size is None:
        filter_width = filter_height = int(np.ceil(radius)*2+1)
    elif hasattr(filter_size, "__getitem__"):
        filter_width, filter_height = filter_size
    else:
        filter_width = filter_height = filter_size
    assert (filter_height>=2*radius+1) and (filter_width >= 2*radius+1)
    filter_zero_x = filter_width/2
    filter_zero_y = filter_height / 2
    angles=np.linspace(0,np.pi*2,nb_samples+1)[:-1]
    filter_bank=np.zeros([nb_samples, filter_width, filter_height])
    for n in range(len(angles)):
        angle=angles[n]
        x=np.cos(angle)*radius
        y=np.sin(angle)*radius
        if x == np.round(x):
            x_left=int(x)
            x_left_coef=1.0
            x_right=0
            x_right_coef=0.0
        else:
            x_left=int(np.floor(x))
            x_right_coef=x-np.floor(x)
            x_right = x_left + 1
            x_left_coef=1-x_right_coef

        if y == np.round(y):
            y_top=int(y)
            y_top_coef=1.0
            y_bottom=0
            y_bottom_coef=0.0
        else:
            y_top=int(np.floor(y))
            y_bottom_coef=y-np.floor(y)
            y_bottom = y_top + 1
            y_top_coef=1-y_bottom_coef

        coef_sum = x_left_coef * y_top_coef + x_left_coef * y_bottom_coef + x_right_coef*y_top_coef + x_right_coef * y_bottom_coef

        filter_bank[n, filter_zero_y+y_top, filter_zero_x+x_left]+= x_left_coef*y_top_coef/coef_sum
        filter_bank[n, filter_zero_y+y_bottom, filter_zero_x+x_left]+= x_left_coef * y_bottom_coef/coef_sum
        filter_bank[n, filter_zero_y+y_top, filter_zero_x+x_right]+=x_right_coef*y_top_coef/coef_sum
        filter_bank[n, filter_zero_y+y_bottom, filter_zero_x+x_right]+= x_right_coef * y_bottom_coef/coef_sum

    filter_bank[:,filter_height/2,filter_width/2]=-1
    filter_bank=filter_bank.reshape([nb_samples,1,filter_width,filter_height])
    return torch.Tensor(filter_bank)


class LBPAlphabet(torch.nn.Module):
    def __init__(self,input_channels):
        super(LBPAlphabet,self).__init__()
        self.W ,self.b = LBPAlphabet.create_binary_alphabet_filter_bank(input_channels)

    def forward(self,x):
        x = torch.nn.functional.conv2d(x, weight=self.W, groups=1)+self.b[None,:,None,None]
        return torch.relu(x)

    @staticmethod
    def create_binary_alphabet_filter_bank(input_channels):
        W = torch.zeros([input_channels,2**input_channels])
        words = torch.arange(2**input_channels, dtype=torch.int32)
        b=[]
        W=[]
        for word in words:
            bit_string=[word%2]
            for pow in range(1,8):
                bit_string.append(((word/2**pow)%2))
            W.append([1.0 if n > 0 else float(-input_channels) for n in bit_string])
            b.append(1.0 - sum(bit_string))
        W = torch.Tensor(W).view(2**input_channels, input_channels, 1, 1)
        b=torch.Tensor(b)
        print repr(W)
        print repr(b)
        #W = torch.Tensor(W).reshape(words, input_channels, 1, 1)
        return W, b

def plot_8_filters(activations,central_activation=None,save_fname=None):
    activations,central_activation=activations.view(activations.size()[1:]),central_activation.view(central_activation.size()[1:])
    fig,ax_list=plt.subplots(3,3)
    if central_activation is not None:
        ax_list[1][1].imshow(central_activation.detach().numpy()[0,:,:],cmap="gray")
    ax_list[1][2].imshow(activations[0, :, :].detach().numpy(),cmap="gray")
    ax_list[0][2].imshow(activations[1, :, :].detach().numpy(),cmap="gray")
    ax_list[0][1].imshow(activations[2, :, :].detach().numpy(),cmap="gray")
    ax_list[0][0].imshow(activations[3, :, :].detach().numpy(),cmap="gray")
    ax_list[1][0].imshow(activations[4, :, :].detach().numpy(),cmap="gray")
    ax_list[2][0].imshow(activations[5, :, :].detach().numpy(),cmap="gray")
    ax_list[2][1].imshow(activations[6, :, :].detach().numpy(),cmap="gray")
    ax_list[2][2].imshow(activations[7, :, :].detach().numpy(),cmap="gray")
    for n in range(3):
        for k in range(3):
            ax_list[n][k].get_xaxis().set_visible(False)
            ax_list[n][k].get_yaxis().set_visible(False)
    #fig.tight_layout()
    if save_fname is not None:
        fig.savefig(save_fname)


def __createCmapCdict__(N,pattern0=0):
    patternValues=np.cumsum(np.ones((N+1),dtype='uint32'))-1;
    resFloatRed=0.45*np.mod(patternValues,2)+0.325*np.mod(patternValues/8,2)+0.225*np.mod(patternValues/64,2)
    resFloatGreen=0.45*np.mod(patternValues/2,2)+0.325*np.mod(patternValues/16,2)+0.225*np.mod(patternValues/128,2)
    resFloatBlue=0.6*np.mod(patternValues/4,2)+0.4*np.mod(patternValues/32,2)
    x=np.linspace(0,1,N)
    rXY0Y1=np.concatenate((x.reshape(-1,1),resFloatRed[:-1].reshape(-1,1),resFloatRed[:-1].reshape(-1,1)),axis=1)
    #print rXY0Y1
    r=tuple([tuple(line) for line in list(rXY0Y1)])
    gXY0Y1=np.concatenate((x.reshape(-1,1),resFloatGreen[:-1].reshape(-1,1),resFloatGreen[:-1].reshape(-1,1)),axis=1)
    g=tuple([tuple(line) for line in gXY0Y1])
    bXY0Y1=np.concatenate((x.reshape(-1,1),resFloatBlue[:-1].reshape(-1,1),resFloatBlue[:-1].reshape(-1,1)),axis=1)
    b=tuple([tuple(line) for line in bXY0Y1])
    cDict={'red':r, 'green':g,'blue':b}
    cMap=matplotlib.colors.LinearSegmentedColormap('d%d'%N,cDict)
    plt.register_cmap(cmap=cMap)

__createCmapCdict__(256)



def plot_256_filters(activations,save_fname=None):
    activations=activations[0,:,:,:]
    fig,ax_list=plt.subplots(12,22)
    counter=0
    for n in range(12):
        for k in range(22):
            if counter<256:
                ax_list[n][k].title.set_text("P:{}".format(counter))
                ax_list[n][k].title.set_size(5)
                ax_list[n][k].imshow(activations[counter, :, :].detach().numpy(), cmap="gray",vmin=0.0, vmax=1.0)
            ax_list[n][k].get_xaxis().set_visible(False)
            ax_list[n][k].get_yaxis().set_visible(False)
            counter+=1
    ax_list[11][21].imshow(activations.detach().numpy().sum(axis=1), cmap="gray")
    fig,ax=plt.subplots()#.view((256,1)+activations.size()[2:])))
    activations=activations.detach().unsqueeze(1)
    _,labels=alphabet_activations[0,:,:,:].max(dim=0)
    plt.imshow(labels.detach().numpy(),cmap="d256")
    plt.colorbar()
    plt.show()
    #image=torchvision.utils.make_grid(activations)
    print repr(image.size())
    cv2.imwrite("/tmp/1.png",image.transpose(1,0).transpose(2,1).numpy())

    if save_fname is not None:
        fig.savefig(save_fname)
    #plt.colorbar()
    #plt.tight_layout()



if __name__=="__main__":
    #torch.relu(torch.nn.functional.conv2d(img, W, groups=1) + b.view(1, 4, 1, 1))
    cv_image=cv2.cvtColor(cv2.imread("../tmp_img/0.jpg",cv2.IMREAD_COLOR)[200:400,200:500,:],cv2.COLOR_BGR2RGB)/255.0
    #plt.imshow(cv_image);plt.show()
    #bin_image = (image<filters.threshold_otsu(image)).astype("float")
    #image = torch.Tensor(np.transpose(cv_image,(2,0,1))[None,:,:,:])
    #print image.size()
    #image=image.view(1,3,200,300)
    image = torch.zeros(1,3,400,300)
    image[0,0, :200 ,:] = torch.Tensor(np.abs(np.linspace(-2+0j,2+0j,200)[:,None]+np.linspace(0-2j,0+2j,300)[None,:]))
    image[0, 1, :200, :] = torch.Tensor(np.abs(np.linspace(-2 + 0j, 2 + 0j, 200)[:, None] + np.linspace(0 - 2j, 0 + 2j, 300)[None, :]))
    image[0, 2, :200, :] = torch.Tensor(np.abs(np.linspace(-2 + 0j, 2 + 0j, 200)[:, None] + np.linspace(0 - 2j, 0 + 2j, 300)[None, :]))
    print image.max(),image.min()
    image=image.max()-image
    image[:, :, :20, :] = 0
    image[:, :, :, :20] = 0
    image[:, :, :, 480:] = 0
    image[:, :, 180:, :] = 0
    print(image.min())
    image=image - image.min()
    image=(image-image.min())/(image.max()-(image.min()+.00000000001))
    #image = torch.zeros(1, 3, 200, 300)
    image[0, :, 250:-50, 50:-50]=.5
    image[0, :, 280:-80, 80:-80]=1
    lbp_layer = OffsetRadialFeatures(3,8,3)
    activations, central_activation = lbp_layer(image)
    alphabet_layer = LBPAlphabet(8)
    alphabet_activations = alphabet_layer(activations*10)
    plot_8_filters(activations[:,:,3:-3,3:-3], central_activation=central_activation)
    plot_256_filters(alphabet_activations*100,save_fname="/tmp/256.pdf")
    print "Alphabet Sum:", activations[0,:,:,:].sum(dim=2).sum(dim=1).to(torch.int32)
    print "Alphabet Max:", activations.view(-1).max()

    print "Alphabet Sum:", alphabet_activations[0,:,:,:].sum(dim=2).sum(dim=1).to(torch.int32)
    print "Alphabet Max:", alphabet_activations.view(-1).max()
    plt.show()
