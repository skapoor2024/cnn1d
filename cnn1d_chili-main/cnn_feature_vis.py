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

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def data_1D(f):
    sed = 100
    d_asta = pd.read_csv(f,encoding='utf-8')
    d_asta = d_asta.loc[:, ~d_asta.columns.str.contains('^Unnamed')]
    #col = [str(x) for x in range(0,151)]+[str(x) for x in range(751,801)]+['Sample ID','Cap']
    #col = ['Sample ID','Cap','800']
    #d_asta = df.drop(columns=col)
    #d_asta = d_asta.dropna(subset=['ASTA'])
    #d_asta['ASTA'] = d_asta['ASTA'].round()
    #d_asta = d_asta[d_asta['ASTA']>=50]
    #d_asta = d_asta[d_asta['ASTA']<120]

    X_val = np.array(d_asta.drop(columns = ['ASTA']))
    Y_val = np.array(d_asta['ASTA'])

    X_val = savitzky(X_val,11,2)
    X_val = MSC(X_val)
    X_val = detrending(X_val)
    X_val = derivate_first(X_val,11,2)


    return X_val[61],Y_val[61]

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using'+device+'device')
    man_seed = 100
    torch.manual_seed(man_seed)
    model = D1CNN()
    model.load_state_dict(torch.load('mdl_asta_40_120_lds_aug/model_asta_40_120_lds_dataaug_5_20220309_182454_659.pth'))
    model.to(device)
    model.to(torch.float32)
    loss = weighted_l1_loss
    file_name = '/home/amc012/Desktop/chili-specx/test_chili_asta.csv'
    x_val,y_val = data_1D(file_name)
    fig_1 = plt.figure(figsize =(18,9))
    ax = fig_1.add_subplot(1,1,1)
    ax.plot(x_val.transpose())
    ax.title.set_text('Original Data')
    model.eval()
    
    with torch.no_grad():

        v_x_dt = torch.from_numpy(x_val)
        v_x_dt = torch.unsqueeze(v_x_dt,dim=0)
        v_x_dt = torch.unsqueeze(v_x_dt,dim=0)
        
        v_x_dt = Variable(v_x_dt,requires_grad=False).to(device,dtype = torch.float32)
        print(v_x_dt.shape)
        v_y_dt = torch.from_numpy(np.array(y_val))
        v_y_dt = Variable(v_y_dt,requires_grad = False).to(device,dtype = torch.float32)
        v_yhat = model.forward(v_x_dt)
        v_yhat = torch.squeeze(v_yhat)
        v_yhat = torch.round(v_yhat)
        v_loss = loss(v_yhat,v_y_dt).item()

        b_yhat = v_yhat.to('cpu')
        b_y_t_1 = v_y_dt.to('cpu')
        #r2_score = sklearn.metrics.r2_score(b_y_t_1,b_yhat)
        
    
        v_x_dt.detach()
        v_y_dt.detach()
        v_yhat.detach()
        
    print("\n the MAE is %.4f\n"%(v_loss)),#r2_score))
    
    #ttt = abs(b_yhat - b_y_t_1)
    
    
    #fig_4 = plt.figure(figsize =(18,9))
    #ax4 = fig_4.add_subplot(1,2,1)
    #ax5 = fig_4.add_subplot(1,2,2)
    #ax4.title.set_text('Scatter plot for best epoch')
    #ax5.title.set_text('MAE for each sample')
    #ax5.plot(b_y_t_1,ttt,marker ='o',linestyle='')
    #ax4.scatter(b_y_t_1,b_yhat,c='red',edgecolors='k')
    #ax4.plot(b_y_t_1,b_y_t_1,c='blue',linewidth = 1)
    #fig_4.show()
    
    model_weights =[]
    conv_layers = []

    model_children = list(model.children())
    counter = 0

    #print(model_children)

    for i in range(len(model_children)):        

        if type(model_children[i])==nn.Sequential:
            for child in model_children[i].children():
                if type(child) == nn.Conv1d:
                    counter +=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
    
    print(f" Total convolution layers:{counter}")
    print("Conv_layers")
    
    test_x = v_x_dt
    outputs =[]
    names = []
    for layer in conv_layers[0:]:
        test_x = layer(test_x)
        outputs.append(test_x)
        names.append(str(layer))
    
    print(len(outputs))

    prcess= []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale/feature_map.shape[0]
        prcess.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(18, 9))
    for i in range(len(prcess)):
        a = fig.add_subplot(4, 2, 2*(i+1))
        imgplot = plt.plot(prcess[i])
        #a.axis("off")
        a.set_title(names[i].split('(')[0]+'layer'+str(i+1), fontsize=10)
        a_2 = fig.add_subplot(4, 2, (2*i)+1)
        plt.plot(x_val.transpose())
        #a.axis("off")
        a_2.set_title('input spectra', fontsize=10)
        
    
    plt.show()
    plt.savefig(str('feature_maps_asta.jpg'), bbox_inches='tight')
        
        
    
    #y_data = pd.DataFrame()
    #y_data['actual'] = b_y_t_1
    #y_data['predcted'] = b_yhat
    #y_data.to_csv('chili_val_test_data.csv')
    
if __name__ == '__main__':
    run()
