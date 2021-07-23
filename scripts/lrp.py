import cv2
import torch
import torch.nn as nn
import torchvision
import numpy
import keras
import copy
import matplotlib.pyplot as plt
import glob
from distributed import Client, wait
from dask_jobqueue import SLURMCluster

def heatmap(R,sx,sy, filename):

    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imsave(png_out_dir+filename[34:],R,cmap=my_cmap,vmin=-b,vmax=b)

def toconv(layers):

    newlayers = []

    for i,layer in enumerate(layers):

        if isinstance(layer,nn.Linear):

            newlayer = None

            if i == 0:
                m,n = 512,layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))

            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))

            newlayer.bias = nn.Parameter(layer.bias)

            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers


def newlayer(layer,g):

    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer

def make_LRP(filename):
    img = numpy.array(cv2.imread(filename))[...,::-1]/255.0

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)

    X = (torch.FloatTensor(img[numpy.newaxis].transpose([0,3,1,2])*1) - mean) / std

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    layers = list(model._modules['features']) + toconv(list(model._modules['classifier']))
    L = len(layers)

    A = [X]+[None]*L
    for l in range(L): A[l+1] = layers[l].forward(A[l])


    T = torch.FloatTensor((1.0*(numpy.arange(1000)==483).reshape([1,1000,1,1])))

    R = [None]*L + [(A[-1]*T).data]


    for l in range(1,L)[::-1]:
        
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

            if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l],rho).forward(A[l]))  
            s = (R[l+1]/z).data                                 
            (z*s).sum().backward(); c = A[l].grad                 
            R[l] = (A[l]*c).data                                  
            
        else:
            
            R[l] = R[l+1]

    for i,l in enumerate([1]):
        heatmap(numpy.array(R[l][0]).sum(axis=0),0.5*i+1.5,0.5*i+1.5, filename)


data_directory = '../clustering/KTLX_4/KLOT_trained/cluster2/*png*'
data_list = glob.glob(data_directory)
png_out_dir = '../clustering/KTLX_4/KLOT_trained/LRP_results/'
i = 0


if __name__ == "__main__":
    
    for filename in data_list:
        make_LRP(filename)
        i+=1
        if i % 10 == 0:
            print("File %d of %d processed" % (i, len(data_list)))
    """
    Cluster = SLURMCluster(processes=6, cores=36, memory='128GB', walltime='15:00')
    Cluster.scale(36)
    client = Client(Cluster)
    print("Waiting for workers...")
    while(len(client.scheduler_info()["workers"]) < 6):
        i = 1
    futures = client.map(make_LRP, data_list)
    wait(futures)
    client.close()"""