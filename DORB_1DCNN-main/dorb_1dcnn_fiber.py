
import numpy as np
import pandas as pd
import random
import torch
import sys
from torch import nn
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics
#from google.colab import files
import io
from scipy import signal as sg
from scipy.signal import savgol_filter
from torch.autograd import Function
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.ndimage import convolve1d
from scipy.signal.windows import triang
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
import sys
import copy

# pynvml import *

b_size = 32 #batch size
np.random.seed(100)

# the 1DCNN model on Asta for regression with added ANN layer and 0.00001 learning rate and with SNV preprocessing with Huberloss and adam optimizer with changed window length =11, polyorder = 2
class D1CNN(nn.Module):
    def __init__(self):
        super(D1CNN,self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('1_conv1d',nn.Conv1d(in_channels=1,out_channels=32,kernel_size=21))
        self.features.add_module('1_relu',nn.ReLU())            
        self.features.add_module('1_batchnorm',nn.BatchNorm1d(32))
        self.features.add_module('1_maxpool',nn.MaxPool1d(kernel_size=2))
    
        self.features.add_module('2_conv1d',nn.Conv1d(in_channels=32,out_channels=64,kernel_size=21))
        self.features.add_module('2_relu',nn.ReLU())
        self.features.add_module('2_batchnorm',nn.BatchNorm1d(64))
        self.features.add_module('2_maxpool',nn.MaxPool1d(kernel_size=3))

        self.features.add_module('3_conv1d',nn.Conv1d(in_channels=64,out_channels=128,kernel_size=21))
        self.features.add_module('3_relu',nn.ReLU())
        self.features.add_module('3_batchnorm',nn.BatchNorm1d(128))
        self.features.add_module('3_maxpool',nn.MaxPool1d(kernel_size=3))

        self.features.add_module('4_conv1d',nn.Conv1d(in_channels=128,out_channels=256,kernel_size=21))
        self.features.add_module('4_relu',nn.ReLU())
        self.features.add_module('4_batchnorm',nn.BatchNorm1d(256))
        self.features.add_module('4_maxpool',nn.MaxPool1d(kernel_size=3))
        
        self.features.add_module('drop_1',nn.Dropout(p=0.4))
    
        self.regress = nn.Sequential()
        self.regress.add_module('an_1',nn.Linear(1024,512))
        self.regress.add_module('an_Relu_1',nn.ReLU())
        self.regress.add_module('an_drop_1',nn.Dropout(p=0.2))
        self.regress.add_module('an_2',nn.Linear(512,512))
        self.regress.add_module('an_Relu_2',nn.ReLU())
        self.regress.add_module('an_drop_2',nn.Dropout(p=0.2))
        self.regress.add_module('an_3',nn.Linear(512,128))
        self.regress.add_module('an_Relu_3',nn.ReLU())
        self.regress.add_module('an_drop_3',nn.Dropout(p=0.2))
        self.regress.add_module('an_4',nn.Linear(128,128))
        self.regress.add_module('an_Relu_4',nn.ReLU())
        self.regress.add_module('an_drop_4',nn.Dropout(p=0.2))
        self.regress.add_module('an_5',nn.Linear(128,64))
        self.regress.add_module('an_Relu_5',nn.ReLU())
        self.regress.add_module('an_drop_5',nn.Dropout(p=0.2))
        self.regress.add_module('an_6',nn.Linear(64,1))

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x_out = self.regress(x)
        return x_out

# preprocessing functions
def moving_smoothing(input_array, window_length):

    m_rows = len(range(window_length, input_array.shape[1] - window_length))
    m_cols = 2*window_length
    matrix = np.zeros((m_rows,m_cols),dtype=int)
    for j in range(0, len(matrix)):
        k = j+1
        matrix[j] = [x for x in range(k,k+2*window_length)]
    #Smoothing spectra using matrix operations:
    n_cols = m_rows
    newspectra = np.zeros((len(input_array),n_cols))
    for i in range(len(matrix)):
        newspectra[:,i] = np.mean(input_array[: ,matrix[i]],axis=1)
    #Add front and end tails (not smoothed):
    #new_spectra = np.asarray(newspectra)
    fronttail = newspectra[:,:1]
    endtail = newspectra[:,-1:]
    for k in range(1,window_length):
        fronttail=np.append(fronttail,newspectra[:,:1], axis=1)
        endtail = np.append(endtail,newspectra[:,-1:], axis =1)
    data = np.concatenate((fronttail, newspectra, endtail), axis=1)
    return data

def derivate_first(input_array, window_length, polyorder):
    der1 = savgol_filter(input_array, window_length, polyorder,deriv = 1)
    return der1

def MSC(input_array, reference = None):
    ''' Perform Multiplicative scatter correction'''

    #Mean correction
    for i in range(input_array.shape[0]):
        input_array[i,:] -= input_array[i,:].mean()

    # Get the reference spectrum. If not given, estimate from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_array, axis=0)
    else:
        ref = reference

    # Define a new data matrix and populate it with the corrected data    
    output_data = np.zeros_like(input_array)
    for i in range(input_array.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_array[i,:], 1, full=True)
        # Apply correction
        output_data[i,:] = (input_array[i,:] - fit[0][1]) / fit[0][0] 
    #print(fit)
    return output_data

def detrending(input_array):
    X = np.arange(input_array.shape[1])
    base = np.zeros((len(input_array),len(X)))
    for i in range(len(input_array)):
        c = np.polyfit(X, input_array[i], 2)
        base[i] = np.polyval(c, X)
    #Baseline removal
    base_remove = input_array - base
    return base_remove

def savitzky(input_array, window_length, polyorder):
    savgol = savgol_filter(input_array, window_length, polyorder)
    return savgol

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def get_weights(l):
    lds_kernel_window = get_lds_kernel_window('gaussian',11, 2)
    smoothed_value = convolve1d(np.asarray(l),weights = lds_kernel_window,mode='constant')
    weights = [np.float32(1 / x) for x in smoothed_value]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

def dataaugment(x, betashift = 0.05, slopeshift = 0.05,multishift = 0.05):
    #Shift of baseline
    #calculate arrays
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x

#to divide data into test train and valid
def data_1D(f_train,f_test):
    sed = 100
    df_train = pd.read_csv(f_train,encoding='utf-8')
    df_test = pd.read_csv(f_test,encoding = 'utf-8')
    df_train= df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
    df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
    
    X_train = np.array(df_train.drop(columns = ['fiber']))
    Y_train = np.array(df_train['fiber'])
    X_val = np.array(df_test.drop(columns = ['fiber']))
    Y_val = np.array(df_test['fiber'])

    r2 = df_train['fiber'].max()+0.1
    r1 = df_train['fiber'].min()
    y = np.zeros((int(r2-r1))*10,dtype=np.int32)
    labels = df_train['fiber'].values
    for label in labels:
        y[round((label - r1)*10)]+=1


    X_train_aug = np.repeat(X_train,repeats = 9, axis = 0)
    shift = .1*np.std(X_train)
    X_train_aug = dataaugment(X_train_aug,betashift = shift,slopeshift = shift/2,multishift = shift)
    X_train = np.concatenate((X_train,X_train_aug),axis = 0)

    Y_train_aug = np.repeat(Y_train,repeats = 9 ,axis = 0)
    Y_train = np.concatenate((Y_train,Y_train_aug),axis = 0)

    idx = np.random.randint(X_train.shape[0], size = X_train.shape[0]//b_size *b_size)
    X_train = X_train[idx,:]
    Y_train = Y_train[idx]

    X_train = savitzky(X_train,11,2)
    X_train = MSC(X_train)
    X_train = detrending(X_train)
    X_train = derivate_first(X_train,11,2)

    X_val = savitzky(X_val,11,2)
    X_val = MSC(X_val)
    X_val = detrending(X_val)
    X_val = derivate_first(X_val,11,2)

    return X_train,Y_train,X_val,Y_val,get_weights((y)*9)

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using'+device+'device')
    man_seed = 100
    torch.manual_seed(man_seed)
    n_epoch = 1000
    model = D1CNN().to(device)
    model.to(torch.float32)
    optimizer = optim.Adam(model.parameters(),lr = 0.000001)
    loss = weighted_l1_loss
    f_train = '/home/amc012/Desktop/chili-specx/dorb_train_2.csv'
    f_test = '/home/amc012/Desktop/chili-specx/dorb_test_2.csv'
    
    x_train,y_train,x_val,y_val,wt = data_1D(f_train,f_test)
    best_v_loss = 1_000_000
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    T_L=[] # to store training loss
    V_L=[]# to store validation loss
    R2_L = [] # to show the R2 score
    
    for e in range(n_epoch):
    
        t_let = 0.0
        t_loss = 0.0
        optimizer.zero_grad()

        #dataset prep for training
        seed = e+1
        #x_train,x_val,y_train,y_val = train_test_split(x_t,y_t,test_size = 0.111,random_state = seed)
        x_train,y_train = shuffle(x_train,y_train,random_state=seed)
        l = x_train.shape[0]//b_size
        w = x_train.shape[1]
        z_x = np.empty((l,b_size,w))
        z_y_1 = np.empty((l,b_size))
        j=0
        for i in range(0,y_train.shape[0],b_size):
            z_x[j] = x_train[i:i+b_size,:]
            z_y_1[j] = y_train[i:i+b_size]
            j = j+1
        
        model.train(mode=True)

        #training on batch data
        t = z_x.shape[0]
        for i in range(t):

            optimizer.zero_grad()
            
            t_x_dt = torch.from_numpy(z_x[i])
            t_x_dt = torch.unsqueeze(t_x_dt,1)
            t_x_dt = Variable(t_x_dt,requires_grad=False).to(device,dtype = torch.float32)
            
            t_yhat = model.forward(t_x_dt)
            t_yhat = torch.squeeze(t_yhat)

            t_y_1_org = torch.from_numpy(z_y_1[i])
            t_y_1_org = Variable(t_y_1_org,requires_grad=False).to(device,dtype = torch.float32)

            weights = torch.tensor([wt[round((x-7.5)*10)] for x in z_y_1[i] ]).to(device=device)

            err = loss(t_yhat,t_y_1_org,weights)

            err.backward()
            optimizer.step()
            t_loss += float(err.item())
            t_let = t_loss/(i+1)

            t_x_dt.detach()
            t_y_1_org.detach()
            t_yhat.detach()
            
            model_wts = model.state_dict()

            #print("\nThe training loss for epoch %d for batch %d//%d is %.4f"%((e+1),(i+1),t,t_let))

        model.train(mode=False)
        model.eval()

        with torch.no_grad():
        
            T_L.append(t_let)

            v_x_dt = torch.from_numpy(x_val)
            v_x_dt = torch.unsqueeze(v_x_dt,dim=1)
            v_x_dt = Variable(v_x_dt,requires_grad=False).to(device,dtype = torch.float32)
            v_y_dt = torch.from_numpy(y_val)
            v_y_dt = Variable(v_y_dt,requires_grad = False).to(device,dtype = torch.float32)
            v_yhat = model.forward(v_x_dt)
            v_yhat = torch.squeeze(v_yhat)
            #v_yhat = torch.round(v_yhat)
            v_loss = loss(v_yhat,v_y_dt).item()

            V_L.append(v_loss)
            v_y_1 = v_y_dt.to('cpu')
            v_yhat_1 = v_yhat.to('cpu')
            r2_score = sklearn.metrics.r2_score(v_y_1,v_yhat_1)
            R2_L.append(r2_score)
            if e%50 == 0:
                print('\n the validation loss for the epoch %d with loss %.4f is %.4f'%(e+1,t_let,v_loss))
                print('\n the r2 score for the epoch %d with loss %.4f is %.4f'%(e+1,t_let,r2_score))

            if v_loss<best_v_loss:
                best_v_loss = v_loss
                model_path = 'mdl_dorb_1/mdl_dorb_fiber_8_15_wr_{}_{}'.format(timestamp,e+1)
                dir_path = model_path+'.pth'
                best_e = e+1
                best_r2_score = r2_score
                b_yhat = v_yhat_1
                b_y_t_1 = v_y_1
                torch.save(model_wts,dir_path)
                

            v_x_dt.detach()
            v_y_dt.detach()
            v_yhat.detach()


    print('the best validation loss is %.4f at epoch %d'%(best_v_loss,best_e))
    print('the best r2 score is %.4f at epoch %d'%(best_r2_score,best_e))
    fig = plt.figure(figsize=(18,9))
    fig_2 = plt.figure(figsize =(18,9))
    fig_3 = plt.figure(figsize =(18,9))
    fig_4 = plt.figure(figsize =(18,9))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = fig_2.add_subplot(1,1,1)
    ax3 = fig_3.add_subplot(1,1,1)
    ax4 = fig_4.add_subplot(1,2,1)
    ax5 = fig_4.add_subplot(1,2,2)
    x_ax = [_ for _ in range(1,n_epoch+1)]
    
    ax1.title.set_text('Training Loss')
    ax2.title.set_text('Validation loss')
    ax3.title.set_text('r2 score')
    ax4.title.set_text('Scatter plot for best epoch')
    ax5.title.set_text('MAE for each sample')
    ax1.plot(x_ax,T_L,color = 'red')
    ax2.plot(x_ax,V_L,color = 'green')
    ax3.plot(x_ax,R2_L,color= 'blue')
    ttt = abs(b_yhat - b_y_t_1)
    ax5.plot(b_y_t_1,ttt,marker ='o',linestyle='')
    ax4.scatter(b_y_t_1,b_yhat,c='red',edgecolors='k')
    ax4.plot(b_y_t_1,b_y_t_1,c='blue',linewidth = 1)


if __name__ == '__main__':
    run()
