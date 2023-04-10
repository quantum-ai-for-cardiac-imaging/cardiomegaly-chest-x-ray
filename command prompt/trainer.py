#!/usr/bin/env python
# coding: utf-8

# #  Hybrid Classical-Quantum Transfer Learning for Cardiomegaly Detection in Chest X-Rays 

# ## load data

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import random
import os
import copy
        
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from qiskit.utils import algorithm_globals
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDense
#get_ipython().run_line_magic('matplotlib', 'inline')

parser = argparse.ArgumentParser(description='trainer')
parser.add_argument('model', type=str, help='(densenet121_model|alexnet_model|PL_qnn_model|Qiskit_easy_2_qnn_model|Qiskit_easy_4_qnn_model)')
parser.add_argument('--init_step', type=int, help='pre-train step')
parser.add_argument('--train_epochs', type=int, help='training step')
parser.add_argument('--q_num', type=int, default=4, help='training step')
args = parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    algorithm_globals.random_seed = seed

seed_everything(123)
    
from watermark import watermark
#get_ipython().run_line_magic('reload_ext', 'watermark')
#get_ipython().run_line_magic('watermark', '')
#get_ipython().run_line_magic('watermark', '--iversions')


# In[2]:


import time
n_qubits = args.q_num                     # Number of qubits.
q_depth = 6                      # Depth of the quantum circuit (number of variational layers).
max_layers = 15                  # Keep 15 even if not all are used.
q_delta = 0.01                   # Initial spread of random quantum weights.

step = 10e-4                     # Learning rate.
weight_decay = 10e-4             # Weight_decay for learning rate.
batch_size = 8                   # Number of samples for each training step.
init_epochs = args.init_step     # Number of init epochs.
train_epochs = args.train_epochs # Number of training epochs.
step_size= 2                     # Learning rate changing epochs.
gamma_lr_scheduler = 0.3         # Learning rate reduction applied every step_size epochs.  


start_time = time.time()         # Start of the computation timer


# In[3]:


from tqdm import tqdm 
import time
start = time.time()
img_size = 256

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.CenterCrop((224,224)),
        #transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


data_dir = 'chexpert-corrected/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                     data_transforms[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Initialize dataloader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                  batch_size=batch_size, shuffle=True) for x in ['train', 'val']}

# function to plot images
def imshow(inp, title=None):
    """Display image from tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # We apply the inverse of the initial normalization operation.
    if transforms.Normalize in data_transforms['val'].transforms:
        norm = tranform_start[trans_list.index(transforms.Normalize)]
        mean = np.array(norm.mean)#np.array([0.485, 0.456, 0.406])
        std = np.array(norm.std)#np.array([0.229, 0.224, 0.225])
        img = std * input_tensor + mean  
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

own_elapsed = time.time() - start
print("Time elapsed: ", own_elapsed)


# In[4]:


# Get a batch of training data
inputs, classes = next(iter(dataloaders['val']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


# In[5]:


#resize image
import cv2
import PIL
from glob import glob

def read_img(img_path,tranform_):
    img = cv2.imread(img_path)
    transform = transforms.Compose(tranform_)
    img = transform(PIL.Image.fromarray(img))
    return img

print('Scans found:', len(image_datasets['train'].imgs)+ len(image_datasets['val'].imgs))


# ### GradCam

# In[6]:


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50,resnext50_32x4d,resnet18,densenet161,wide_resnet50_2, densenet121, densenet161
from PIL import Image


# ### pre-process dataset
# generate image from address

# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#device = 'cpu' #qiskit don't support gpu on windows


# In[8]:


from tqdm import tqdm 
train_img = []
train_ids = []
train_y = []

img_size=256
tranform_ = [transforms.Resize((img_size, img_size)), 
             transforms.CenterCrop((224,224)),
             #transforms.RandomRotation(30),
             #transforms.RandomHorizontalFlip(p=0.5),
             #transforms.RandomAutocontrast(p=0.5),
             #transforms.ToTensor(),
             #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]

train_img = []
train_ids = []
train_y = []

for img_path in tqdm(image_datasets["train"].imgs):
    train_img.append(read_img(img_path[0],tranform_))
    train_ids.append(img_path[0].split("\\")[2])
    train_y.append(img_path[1])
    
#valid_img = []
#valid_ids = []
#valid_y = []
#
#for img_path in tqdm(image_datasets["val"].imgs):
#    valid_img.append(read_img(img_path[0],tranform_))
#    valid_ids.append(img_path[0].split("\\")[2])
#    valid_y.append(img_path[1])
    
test_img = []
test_ids = []
test_y = []
for img_path in tqdm(image_datasets["val"].imgs):
    test_img.append(read_img(img_path[0],tranform_))
    test_ids.append(img_path[0].split("\\")[2])
    test_y.append(img_path[1])


# In[9]:


#qamp_IMAGE_DIR = "qamp"

import time
start = time.time()

# * better way

tranform_ = [#transforms.Resize((img_size, img_size)), 
             #transforms.CenterCrop((224,224)),
             #transforms.RandomRotation(30),
             #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
             #transforms.RandomAutocontrast(p=1.0),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
transform = transforms.Compose(tranform_)

train_imgg = []
for img in tqdm(train_img):
    img = transform(img)
    train_imgg.append(img)
##train_ds=torch.tensor(np.array(train_img), device=device).float()
train_y = torch.tensor(train_y, device=device).float()
train_ds = TensorDataset(torch.stack(train_imgg).to(device), train_y)
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
train_size = len(train_dataloader)

tranform_ = [#transforms.Resize((img_size, img_size)), 
             #transforms.CenterCrop((224,224)),
             #transforms.RandomRotation(30),
             #transforms.RandomHorizontalFlip(p=0.5),
             #transforms.RandomAutocontrast(p=1.0),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
transform = transforms.Compose(tranform_)

#valid_imgg=[]
#for img in tqdm(valid_img):
#    img = transform(img)
#    valid_imgg.append(img)
###valid_ds=torch.tensor(np.array(valid_img), device=device).float()
#valid_y = torch.tensor(valid_df['Cardiomegaly'].values, device=device).float()
#valid_ds = TensorDataset(torch.stack(valid_imgg).to(device), valid_y)
#valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=2, shuffle=True)
#valid_size = len(valid_dataloader)

test_img_ = []
for img in tqdm(test_img):
    img = transform(img)
    test_img_.append(img)
#test_ds=torch.tensor(np.array(test_img_), device=device).float()
test_y = torch.tensor(test_y, device=device).float()
test_ds = TensorDataset(torch.stack(test_img_).to(device), test_y)
test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)#, sampler= sampler)


own_elapsed = time.time() - start
print("Time elapsed: ", own_elapsed)


# ## Neural Network

# In[10]:


# install instruction - https://www.delftstack.com/howto/python/python-graphviz-executables-are-not-found/
#from torchviz import make_dot
#
#model = resnet18()
#batch = next(iter(train_dataloader))[0]
##x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)#8, 3, 224, 224
#yhat = model(batch) # Give dummy batch to forward().
#make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")


# In[11]:


from math import pi
from scipy.special import logsumexp
import numpy as np


# This class is used to calculate the effective dimension of a model (classical or quantum)
# It implicitly computed the normalised Fisher information (which is called fhat) and then computes the eff dimension


class EffectiveDimension:
    def __init__(self, model, num_thetas, num_inputs):
        """
        Computes the effective dimension for a parameterised model.
        :param model: class instance
        :param num_thetas: int, number of parameter sets to include
        :param num_inputs: int, number of input samples to include
        """
        self.model = model
        self.d = model.d
        self.num_thetas = num_thetas
        self.num_inputs = num_inputs
        # Stack data together and combine parameter sets to make calcs more efficient
        rep_range = np.tile(np.array([num_inputs]), num_thetas)
        params = np.random.uniform(self.model.thetamin, self.model.thetamax, size=(self.num_thetas, model.d))
        self.params = np.repeat(params, repeats=rep_range, axis=0)
        x = np.random.normal(0, 1, size=(self.num_inputs, self.model.inputsize))
        self.x = np.tile(x, (self.num_thetas, 1))

    def get_fhat(self):
        """
        :return: ndarray, f_hat values of size (num_inputs, d, d)
        """
        grads = self.model.get_gradient(params=self.params, x=self.x)  # get gradients, dp_theta
        output = self.model.forward(params=self.params, x=self.x)  # get model output
        fishers = self.model.get_fisher(gradients=grads, model_output=output)
        fisher_trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
        # average the fishers over the num_inputs to get the empirical fishers
        fisher = np.average(np.reshape(fishers, (self.num_thetas, self.num_inputs, self.d, self.d)), axis=1)
        f_hat = self.d * fisher / fisher_trace  # calculate f_hats for all the empirical fishers
        return f_hat, fisher_trace

    def eff_dim(self, f_hat, n):
        """
        Compute the effective dimension.
        :param f_hat: ndarray
        :param n: list, used to represent number of data samples available as per the effective dimension calc
        :return: list, effective dimension for each n
        """
        effective_dim = []
        for ns in n:
            Fhat = f_hat * ns / (2 * pi * np.log(ns))
            one_plus_F = np.eye(self.d) + Fhat
            det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
            r = det / 2  # divide by 2 because of sqrt
            effective_dim.append(2 * (logsumexp(r) - np.log(self.num_thetas)) / np.log(ns / (2 * pi * np.log(ns))))
        return effective_dim


# ### Quantum layer

# In[12]:


import pennylane as qml
dev = qml.device('default.qubit', wires=n_qubits)


# In[13]:


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates. 
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
        
def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis. 
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    #CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT  
    for i in range(0, nqubits - 1, 2): #loop over even indices: i=0,2,...N-2  
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2): #loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


# In[14]:


@qml.qnode(dev, interface='torch') 
def q_net(q_in, q_weights_flat):
        
        # Reshape weights
        q_weights = q_weights_flat.reshape(max_layers, n_qubits)
        
        # Start from state |+> , unbiased w.r.t. |0> and |1>
        H_layer(n_qubits)   
        
        # Embed features in the quantum node
        RY_layer(q_in)
        
       
        # Sequence of trainable variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k + 1])

        # Expectation values in the Z basis
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]


# ### training function

# In[15]:


import copy
from torchsummary import summary as quick_sum # this is buggy, but light
from torchinfo import summary # this is good, but take a lot of memory
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, auc, roc_curve, accuracy_score,confusion_matrix,classification_report
save_result = True
if save_result == True:
    # create folder and save the result
    time_str = time.strftime("%Y%m%d-%H%M%S")
    result_OUT = f"image_result/{args.model}_{args.init_step}_{time_str}"
    os.makedirs(result_OUT, exist_ok=True)

f = open("result.txt", "a")
f.write(f"{args.model}_{args.init_step}_{time_str}\n")
f.close()


# In[16]:


def get_variable_name(variable):
    globals_dict = globals()

    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]


# In[17]:


def conf_matrix(y, y_pred):
    fig, ax =plt.subplots(figsize=(3.5,3.5))
    labels=['No','Yes']
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size":25})
    plt.title('Cardiomegaly?', fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17) 
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test')
    ax.set_xlabel('Predicted')


# In[18]:


def grad_camera(models, target_layers, picture_select, transform_start=None, transform_end=None,name=None):
    """ Train function
    Args:
        models: pytroch model
        target_layers :  target layer of the model
        picture_select (str): select picture from all the image
        transform_start : input picture display
        transform_end : output picture display
    Returns:
        pytorch model
    Raises:
        None
    """
    #target_layers = target_layers
    # need modify yourself
    if transform_start == None:
        tranform_start = [transforms.Resize((img_size, img_size)), 
                        transforms.CenterCrop((224,224)),
                        #transforms.RandomRotation(30),
                        #transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomAutocontrast(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]
        #return tranform_start
    input_tensor = torch.stack([read_img(eval(f'image_datasets["train"].imgs{picture_select}[0]'), tranform_start)])#torch.stack(train_img)[:1,:,:,:]# Create an input tensor image for your model..
    print(input_tensor.numpy()[0].shape)
    y= models(input_tensor.to(device))
    # Note: input_tensor can be a batch tensor with several images!
    
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAMPlusPlus(model=models, target_layers=target_layers)#, use_cuda=False)
    
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor.to(device))[0, :]#, targets=targets)[0, :]
    
    # In this example grayscale_cam has only one image in the batch:
    if transform_end == None:
        trans_list = list(map(type, tranform_start))
        input_tensor = input_tensor.numpy()[0]
        input_tensor = np.transpose(input_tensor, (1,2,0))
        if transforms.Normalize in trans_list:
            #img = np.transpose(input_tensor, (1,2,0))
            # We apply the inverse of the initial normalization operation.
            norm = tranform_start[trans_list.index(transforms.Normalize)]
            mean = np.array(norm.mean)#np.array([0.485, 0.456, 0.406])
            std = np.array(norm.std)#np.array([0.229, 0.224, 0.225])
            img = std * input_tensor + mean  
            input_tensor = np.clip(img, 0, 1)
    #img= read_img(eval(f'list(all_image_paths.values()){picture_select}'), transform_end)
    #img =np.float32(img) / 255
    visualization = show_cam_on_image(input_tensor, grayscale_cam, use_rgb=True)
    
    
    #_, y_prob = torch.max(y, 1)
    y_probb = nn.Sigmoid()(y)[0]
    df = image_datasets["train"].imgs
    plt.title(f'Cardiomegaly?\n true:{eval(f"df{picture_select}[1]")}\n pred[no/yes]:{np.round(y_probb.cpu().detach().numpy(),3)}', fontsize=20)
    #Image.fromarray(visualization, 'RGB')
    plt.imshow(visualization)
    if (save_result == True and name != None): plt.savefig(f"{result_OUT}/{name}_visual_img.png")
    return visualization


# In[19]:


def train_model(model, criterion, optimizer,scheduler, num_epochs, loss_save=None, scheduler_set = ["outside_train"], sig_out = True,transform = True, save = True):
    """ Train function
    Args:
        model : pytroch model
        criterion :  Criterion
        optimizer : optimize
        scheduler : scheduler
        num_epochs (int): Number of epochs
        loss_save (bool): select model save condition, 
            it can be 'None' for best_loss_train, 'True' for best_acc or False for best_loss (default=None)
        scheduler_set (list): The set the scheduler place, 
            it can include ('inside_train', 'outside_train', 'inside_valid', 'outside_valid', outside_loss_valid)
        sig_out (bool): training sigmoid output
        transform (bool): transform each epoch
    Returns:
        pytorch model
    Raises:
        None
    """
    #qamp_IMAGE_DIR = "qamp"
    start = time.time()
    
    #layer_collection = LayerCollection.from_model(model)
    #d = layer_collection.numel()
    #print('d= ', d)
    name = get_variable_name(model)[0]
    print('name:',name)
    
    #train_ds=torch.tensor(np.array(train_img), device=device).float()
    
    
    #own_elapsed = time.time() - start
    #print("loader Time elapsed: ", own_elapsed)
    global train_ds
    global train_dataloader
    #global valid_ds
    #global valid_dataloader
    
    train_loss = []
    train_loss_list = []
    training_loss = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0   # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    opt_rate=[]
    for epoch in range(num_epochs):
        if transform:
            trans_start = time.time()
            tranform_ = [#transforms.Resize((img_size, img_size)), 
                 #transforms.CenterCrop((224,224)),
                 #transforms.RandomRotation(30),
                 #transforms.RandomHorizontalFlip(p=.5),
                 #transforms.RandomVerticalFlip(p=.5),
                 transforms.RandAugment(),
                 #transforms.RandomAdjustSharpness(sharpness_factor=2),
                 #transforms.RandomSolarize(threshold=192.0),
                 #transforms.RandomSolarize(threshold=192.0),
                 #transforms.RandomInvert(),
                 transforms.RandomAutocontrast(p=.5),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            transform = transforms.Compose(tranform_)
            
            train_imgg = []
            for img in train_img:
                img = transform(img)
                train_imgg.append(img)
            
            train_ds = TensorDataset(torch.stack(train_imgg).to(device), train_y)
            train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
            
            #valid_imgg=[]
            #for img in valid_img:
            #    img = transform(img)
            #    valid_imgg.append(img)
            #valid_ds = TensorDataset(torch.stack(valid_imgg).to(device), valid_y)
            #valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=2, shuffle=False)
            
            print("Transform Time elapsed: ", time.time() - trans_start)
        
        train_size = len(train_dataloader)
        #valid_size = len(valid_dataloader)
        
        running_loss = 0.0
        running_corrects = 0
        print(epoch+1,'Training started:')
        #for dataset_size_multiplier in range(5):
        with tqdm(train_dataloader) as t: 
            for index, data in enumerate(t):
                inputs, labels = data[0].to(device).type(torch.float), data[1].to(device).type(torch.long)
                #running_loss = 0.0
                # Set model to training mode
                model.train() 

                # Each epoch has a training and validation phase
                batch_size_ = len(inputs)
                optimizer.zero_grad()

                # Iterate over data.
                #n_batches = dataset_sizes[phase] // batch_size
                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    #nn.sigmoid(outputs)
                    #print(outputs)
                    if len(outputs.shape) == 1:
                        outputs = torch.stack([outputs])
                    if sig_out == True:
                        outputs = nn.Sigmoid()(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)#torch.stack(label_list))
                    loss.backward()
                    optimizer.step()
                    if "inside_train" in scheduler_set:
                        scheduler.step()


                # Print iteration results
                running_loss += loss.item() #*inputs.size(0)#* batch_size_
                train_loss.append(loss.item() )#*inputs.size(0))#*batch_size_)
                batch_corrects = torch.sum(preds == labels.data).item()
                running_corrects += batch_corrects
                if (index+1)%train_size == 0:
                    #print(outputs)
                    #print(preds)
                    print('Train Epoch: {}/{} train loss {:.4f} Acc batch: {:.4f} learning_rate: {:.4f}'.format(epoch + 1, num_epochs, running_loss/train_size, running_corrects/train_ds.tensors[0].size(0),optimizer.state_dict()['param_groups'][0]['lr']),"time:", t)
                    training_loss.append(running_loss/train_size)
        if running_corrects/len(train_ds) > best_acc_train:
            best_acc_train = running_loss
        if running_loss < best_loss_train:
            best_loss_train = running_corrects/len(train_ds)#round(new_train_df.shape[0]*split_size)
        if "outside_train" in scheduler_set:
            scheduler.step()
                
        train_loss_list.append(np.mean(train_loss))
        #opt_rate.append(optimizer.state_dict()['param_groups'][0]['lr'])
    if save_result: np.savetxt(f"{result_OUT}/{name}_training_loss.csv", training_loss)#save
    if save_result: np.savetxt(f"{result_OUT}/{name}_train_loss.csv", train_loss)#save
    if save_result: np.savetxt(f"{result_OUT}/{name}_train_loss_list.csv", train_loss_list)#save
            
    own_elapsed = time.time() - start
    print("\nTime elapsed: ", own_elapsed)
    
    # Print final results 
    #if loss_save != None:
    #    model.load_state_dict(best_model_wts)
    #print('Best test loss: {:.4f} | Best test accuracy: {:.4f}'.format(best_loss, best_acc))
    if len(train_loss_list) > 1:
        plt.rcParams["figure.figsize"] = (5.5, 4)
        plt.title("Training loss against Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.plot(range(0,len(train_loss_list)), train_loss_list, color="blue")
        if save_result == True: plt.savefig(f"{result_OUT}/{name}_train_loss.png")#save pic
        #plt.show()
    if save:
        running_loss = 0.0
        with torch.no_grad():
            model.eval()
            
            y_pred = []
            for index, data in enumerate(tqdm(test_dataloader)):
                model.eval()
                batch_inputs, batch_labels = data[0].to(device).type(torch.float), data[1].to(device).type(torch.long)
                with torch.set_grad_enabled(False):
                    outputs = model(batch_inputs)
                    y_pred.append(outputs[0])
                    #assert isinstance(outputs.item(), float)
                    #print(outputs, batch_labels)
                    #if len(outputs.shape) == 1:
                    #    outputs = torch.stack([outputs])
                    if sig_out == True:
                        outputs = nn.Sigmoid()(outputs)
                    #_, preds = torch.max(outputs, 1)
                    loss    = criterion(outputs, batch_labels)
                #print(f"Step {index} loss: {loss}")
                running_loss += loss.item()
            _, y_pred_prob = torch.max(torch.stack(y_pred), 1)
            y_pred_prob = y_pred_prob.cpu()
            if save_result == True: np.savetxt(f"{result_OUT}/{name}_test_loss.csv", y_pred_prob)#save
            #print(y_pred, y_pred_prob)
            print("\nclassification_report:")
            print(classification_report(test_y.cpu().long(), torch.tensor(y_pred_prob.detach().numpy()),digits=4))
            conf_matrix(test_y.cpu().long(), y_pred_prob)
            if save_result == True: plt.savefig(f"{result_OUT}/{name}_conf_matrix.png")#save pic
            plt.figure(figsize = (5.5, 4))
            fpr, tpr, _ =roc_curve(test_y.cpu().long(), torch.tensor(y_pred_prob.detach().numpy()))
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
            plt.plot([0, 1], [0, 1],'r--')
            plt.title('ROC curve',fontsize=25)
            plt.ylabel('True Positive Rate',fontsize=18)
            plt.xlabel('False Positive Rate',fontsize=18)
            plt.legend(loc = 'lower right', fontsize=24, fancybox=True, shadow=True, frameon=True, handlelength=0)
            if save_result == True: plt.savefig(f"{result_OUT}/{name}_roc.png") #save pic
            #plt.show()
    return model


# ## init_freezer

# In[20]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def n_count_parameters(model):
    return sum(p.numel() for p in model.parameters() if not(p.requires_grad))


# In[21]:


def train_func(cnn_model,init_step, train_step,save_model=save_result):
    criterion = nn.CrossEntropyLoss()
    if init_step > 0:
        print("\nstart init:")
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=step, weight_decay = weight_decay)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cnn_model.parameters()), lr=10e-4, momentum=0.9)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_lr_scheduler)
        cnn_model = train_model(cnn_model.to(device), 
                                criterion, optimizer,exp_lr_scheduler, 
                                init_step,None,["outside_train"], False, True,False)
    
    for param in cnn_model.parameters():
        param.requires_grad = True
    print("\nstart train:")
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=(step*0.3)*(np.floor(init_step/2)) if init_step > 0  else step, weight_decay = weight_decay)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cnn_model.parameters()), lr=3e-4, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_lr_scheduler)
    cnn_model = train_model(cnn_model.to(device), 
                            criterion, optimizer,exp_lr_scheduler, 
                            train_step,None,["outside_train"], False, True)#["outside_train"]
    
    cnn_model_visual = grad_camera(cnn_model, [cnn_model.features[-1]],[0] ,transform_start=None, transform_end=None,name=get_variable_name(cnn_model)[0])
    if save_model:
        torch.save(cnn_model.state_dict(),f"{result_OUT}/{get_variable_name(cnn_model)[0]}.pt")
    return cnn_model


# In[22]:


def prob_loop(sample,models, target_layers, picture_select, transform_start=None, transform_end=None,name=None):
    # need modify yourself
    if transform_start == None:
        tranform_start = [transforms.Resize((img_size, img_size)), 
                        transforms.CenterCrop((224,224)),
                        #transforms.RandomRotation(30),
                        #transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomAutocontrast(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]
    input_tensor = torch.stack([read_img(eval(f'image_datasets[sample].imgs{picture_select}[0]'), tranform_start)])#torch.stack(val_img)[:1,:,:,:]# Create an input tensor image for your model..
    y= models(input_tensor.to(device))
    
    y_probb = nn.Sigmoid()(y)[0]
    return y_probb


def make_df_y_probb(sample,model) :  
    df_y_probb = pd.DataFrame(columns=['image', 'p_yes', 'p_no'])
    df = image_datasets[sample].imgs
    for i in tqdm(range(len(df))):
        image_name = df[i][0][-10:]
        y_probb = prob_loop(sample, model, [model.features[-1]] \
            ,[i] ,transform_start=None, transform_end=None)
        y_probb = y_probb.cpu()
        p_yes = y_probb.detach().numpy()[0]
        p_no = y_probb.detach().numpy()[1]
        row = {'image' : image_name, 'p_yes': p_yes, 'p_no': p_no}
        new_df = pd.DataFrame([row])
        df_y_probb =  pd.concat([df_y_probb, new_df], axis = 0, ignore_index=True)
    return df_y_probb

def plot_ROC(name,y_true, y_score):

    # ROC Curve and AUROC
    plt.figure(0).clf()
    if np.sum(y_true) != 0.:        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = round(roc_auc_score(y_true, y_score), 4)
        plt.plot(fpr,tpr,label=name + ", AUC =" + str(auc))
    if save_result == True: plt.savefig(f"{result_OUT}/{name}_roc.png") #save pic
    plt.legend()
    #plt.show()

def make_csv(model, sample="val",result=True):
    df = make_df_y_probb(sample,model)
    if sample == "val":
        df_list = 361*[1] + 369*[0]
    elif sample == "train":
        df_list = 850*[1] + 856*[0]
    df = df.assign(label = df_list)

    pred_list = np.array(df.p_yes) / (np.array(df.p_yes) + np.array(df.p_no))
    df = df.assign(cmg_pred = pred_list)
    #elif sample == "train":
    #    pred_val_list =( np.array(df.p_yes) / (np.array(df.p_yes) + np.array(df.p_no)) + 0.5).astype('int32')
    #    df = df.assign(pred = pred_val_list)
    
    pred_list =( np.array(df.p_yes) / (np.array(df.p_yes) + np.array(df.p_no)) + 0.5).astype('int32')
    df = df.assign(pred = pred_list)
    
    df = df.rename(columns={"cmg_pred": "prob"})
    if sample == "val":
        df.sort_values(by = 'image')[0:16]
    elif sample == "train":
        df.sort_values(by = 'image')
    if save_result: df.to_csv(f'{result_OUT}/y_probbs_{get_variable_name(model)[0]}_{sample}.csv', index = False)
    
    if result:
        name = get_variable_name(model)[0]
        df_val = pd.read_csv(f'{result_OUT}/y_probbs_{name}_{sample}.csv') 
        prediction = np.array(df_val.prob) 
        name_ROC = f"Cardiomegaly - {name}_{sample}"
        plot_ROC(name_ROC, np.array(df_val.label), prediction)
        print("{sample}:")
        print(classification_report(np.array(df_val.label),  (prediction+ 0.5).astype('int32')))#prediction))
    return df


# ### densenet121
# 

# In[23]:

if args.model == "densenet121_model": 
    densenet121_model = torchvision.models.densenet121(weights= 'DEFAULT')
    for param in densenet121_model.parameters():
        param.requires_grad = False
    densenet121_model.classifier = nn.Sequential(nn.Linear(densenet121_model.classifier.in_features, 512), torch.nn.ReLU(), nn.Linear(512, 2))
    densenet121_model = train_func(densenet121_model, init_epochs, train_epochs)

    for i in ['train','val']:
        exec(f"densenet121_model_{i}=make_csv(densenet121_model,'{i}')")


# ### AlexNet hybird

# In[ ]:

if args.model == "alexnet_model": 
    alexnet_model = torchvision.models.alexnet(weights='DEFAULT')
    for param in alexnet_model.parameters():
        param.requires_grad = False
    alexnet_model.classifier.append(torch.nn.ReLU())
    alexnet_model.classifier.append(nn.Linear(1000, 2))
    #alexnet_model.classifier = nn.Sequential(nn.Linear(cnn_model.classifier[1].in_features, 512), torch.nn.ReLU(), nn.Linear(512, 2))
    alexnet_model = train_func(alexnet_model, init_epochs, train_epochs)
    for i in ['train','val']:
        exec(f"alexnet_model_{i}=make_csv(alexnet_model,'{i}')")


# ### Densenet 121 Pennylane hybird

# In[25]:

if args.model == "PL_qnn_model": 
    class Quantumnet(nn.Module):
        def __init__(self,input_size):
            super().__init__()
            self.pre_net = nn.Linear(input_size, n_qubits)
            self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
            self.post_net = nn.Linear(n_qubits, 2)

        def forward(self, input_features):
            pre_out = self.pre_net(input_features) 
            q_in = torch.tanh(pre_out) * np.pi / 2.0   

            # Apply the quantum circuit to each element of the batch and append to q_out
            q_out = torch.Tensor(0, n_qubits)
            q_out = q_out.to(device)
            for elem in q_in:
                q_out_elem = q_net(elem,self.q_params).float().unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))
            return self.post_net(q_out)

    PL_qnn_model = torchvision.models.densenet121(weights= 'DEFAULT')
    for param in PL_qnn_model.parameters():
        param.requires_grad = False
    PL_qnn_model.classifier = Quantumnet(PL_qnn_model.classifier.in_features)
     #quick_sum(qnn_model,torch.stack(train_imgg).shape[1:],device=device)
    PL_qnn_model= train_func(PL_qnn_model, init_epochs, train_epochs)

    '''
    # Downgrade cuda driver if you meet this error:
    nvrtc: error: failed to open nvrtc-builtins64_117.dll.
      Make sure that nvrtc-builtins64_117.dll is installed correctly.
    '''

    for i in ['train','val']:
        exec(f"PL_qnn_model_{i}=make_csv(PL_qnn_model,'{i}')")


# ### prepare for simulator

# In[26]:

if args.model == "Qiskit_easy_2_qnn_model" or args.model == "Qiskit_easy_4_qnn_model" : 
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap,EfficientSU2, PauliFeatureMap,TwoLocal,ZFeatureMap,NLocal
    from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN, OpflowQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit.utils import QuantumInstance, algorithm_globals
    from qiskit.opflow.gradients import Gradient, Hessian #, NaturalGradient, QFI, 
    from qiskit import QuantumCircuit, transpile, Aer ,  execute, BasicAer, assemble, IBMQ
    from qiskit.circuit import ParameterVector
    from qiskit_machine_learning.neural_networks import EffectiveDimension, LocalEffectiveDimension
    from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
    input_dim = n_qubits


    # In[27]:


    def simulator_train(feature_map, ansatz,effective=None,hardware='CPU',num_layer=0,output_dim=4):
        # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
        qc = feature_map.compose(ansatz)
        simulator = Aer.get_backend('aer_simulator_statevector')
        # detail_explaination: https://qiskit.org/documentation/stubs/qiskit_aer.AerSimulator.html
        simulator.set_options(
            device=hardware,
            max_parallel_threads = 0, #max_parallel_shots
            max_parallel_experiments = 0, #equal 0 means maximum number of experiments that may be executed in parallel equals max_parallel_threads value.
            max_parallel_shots = 1, #equal 1 means disable parallel shot execution.
            statevector_parallel_threshold = input_dim,
            blocking_enable = True,
        )
        qnn = CircuitQNN(
                    circuit=qc,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    interpret=lambda x: x % output_dim,
                    output_shape=output_dim,
                    gradient= None,#Gradient(grad_method='lin_comb'), #param_shift
                    quantum_instance= QuantumInstance(simulator, shots=10),
                    input_gradients=True,
                )
        initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn.num_weights) - 1)
        model1 = TorchConnector(qnn, initial_weights=initial_weights)
        print(model1(torch.tensor(0.1 * (2 * algorithm_globals.random.random(input_dim) - 1))))
        if effective:
            # we can set the total number of input samples and weight samples for random selection

            global_ed = EffectiveDimension(
                qnn=qnn, weight_samples=10, input_samples=10
            )

            # finally, we will define ranges to test different numbers of data, n
            n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
            #n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
            global_eff_dim_0 = global_ed.get_effective_dimension(dataset_size=n[0])

            d = qnn.num_weights

            print("Data size: {}, global effective dimension: {:.4f}".format(n[0], global_eff_dim_0))
            print(
                "Number of weights: {}, normalized effective dimension: {:.4f}".format(d, global_eff_dim_0 / d)
            )

            global_eff_dim_1 = global_ed.get_effective_dimension(dataset_size=n)

            print("Effective dimension: {}".format(global_eff_dim_1))
            print("Number of weights: {}".format(d))

            # plot the normalized effective dimension for the model
            plt.plot(n, np.array(global_eff_dim_1) / d)
            plt.xlabel("Number of data")
            plt.ylabel("Normalized GLOBAL effective dimension")
            if save_result : plt.savefig(f"{result_OUT}/{effective}_effective.png")
            #plt.show()
        class Quantumnet(nn.Module):
            def __init__(self,input_size):
                super().__init__()
                self.pre_net = nn.Linear(input_size, input_dim)
                #self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
                self.qnn = TorchConnector(qnn)#,initial_weights)
                for i in range(num_layer):
                    exec(f"self.qnn{i}= TorchConnector(qnn{i})")
                self.post_net = nn.Linear(output_dim, 2)

            def forward(self, input_features):
                pre_out = self.pre_net(input_features)
                q_in = torch.tanh(pre_out) * np.pi / 2.0   
                q_out = self.qnn(q_in).float()
                #print(q_out)
                for i in range(num_layer):
                    q_out = torch.tanh(q_out) * np.pi / 2.0  
                    q_out = eval(f"self.qnn{i}().float()")

                # Apply the quantum circuit to each element of the batch and append to q_out
                #q_out = torch.Tensor(0, n_qubits)
                #q_out = q_out.to(device)
                return self.post_net(q_out)
        return Quantumnet


    # ### Densenet Qiskit hybird easy qnn
    # same circuit from pennylane, might have better result than [Automated Detection of Alzheimer’s via Hybrid Classical Quantum Neural Networks](https://www.mdpi.com/2079-9292/11/5/721/pdf-vor)

    # In[28]:


    parm = ParameterVector('phi', input_dim)
    param = ParameterVector('deta',q_depth*input_dim)
    feature_map = QuantumCircuit(input_dim)
    feature_map.h(range(input_dim))
    for i,j in enumerate(parm):
        feature_map.ry(j,i)
    #for k in range(feature_map.num_qubits-1):
    #    feature_map.cnot(k,k+1)
    feature_map.barrier()

    feature_map.draw()

    ansatz = QuantumCircuit(input_dim)
    k=0
    for i in range(q_depth):
        for j in range(ansatz.num_qubits-1):
            ansatz.cnot(j,j+1)
        ansatz.barrier()

        for j in range(ansatz.num_qubits):
            ansatz.ry(param[k],j)
            k+=1
        ansatz.barrier()
    print(ansatz.num_parameters)
    feature_map.compose(ansatz).draw()

    #feature_map = ZFeatureMap(input_dim,reps=1)
    #ansatz = RealAmplitudes(input_dim, reps=q_depth,entanglement='linear',skip_final_rotation_layer=True,insert_barriers=True)
    #feature_map.decompose(reps=1).draw()


# #### 2 Dim

# In[29]:

if args.model == "Qiskit_easy_2_qnn_model":
    Quantumnet = simulator_train(feature_map, ansatz,'easy-2',output_dim=2)


    # In[30]:


    Qiskit_easy_2_qnn_model = torchvision.models.densenet121(weights= 'DEFAULT')
    for param in Qiskit_easy_2_qnn_model.parameters():
        param.requires_grad = False
    Qiskit_easy_2_qnn_model.classifier = Quantumnet(Qiskit_easy_2_qnn_model.classifier.in_features)
    Qiskit_easy_2_qnn_model = train_func(Qiskit_easy_2_qnn_model, init_epochs, train_epochs)
    for i in ['train','val']:
        exec(f"Qiskit_easy_2_qnn_model_{i}=make_csv(Qiskit_easy_2_qnn_model,'{i}')")


# #### 4 Dim

# In[31]:

if args.model == "Qiskit_easy_4_qnn_model":

    Quantumnet = simulator_train(feature_map, ansatz,'easy-4',output_dim=4)


    # In[32]:


    Qiskit_easy_4_qnn_model = torchvision.models.densenet121(weights= 'DEFAULT')
    for param in Qiskit_easy_4_qnn_model.parameters():
        param.requires_grad = False
    Qiskit_easy_4_qnn_model.classifier = Quantumnet(Qiskit_easy_4_qnn_model.classifier.in_features)
    Qiskit_easy_4_qnn_model = train_func(Qiskit_easy_4_qnn_model, init_epochs, train_epochs)
    for i in ['train','val']:
        exec(f"Qiskit_easy_4_qnn_model_{i}=make_csv(Qiskit_easy_4_qnn_model,'{i}')")

print("Total_runtime: ", time.time() - start_time)