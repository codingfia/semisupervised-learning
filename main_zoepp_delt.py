# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:31:13 2021

@author: Yingying Wang

modified from the work of "Alfarraj M, AlRegib G. Semisupervised sequence modeling for elastic impedance inversion[J]. Interpretation, 2019, 7(3): SE237-SE249."
"""

import argparse
import numpy as np
import torch
from bruges.filters import wavelets
from os.path import isdir
import os
from models_zoepp import inverse_model, forward_model_zoepp
from torch.utils import data
from functions import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import wget
import hashlib
import scipy.io as sio
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Manual seeds for reproducibility
random_seed=30
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_data(args, test=False):
    #Loading data   
    data_dir = "./data/Marmousi2"
    
    # load the noise-free data
    mat_contents = sio.loadmat(data_dir+'/seismic_data_part_down_VpVsRho_0-2-30.mat')
    seismic_data_org = mat_contents['seismic_data_down']   #原数据格式为 x z  angle
    seismic_data = seismic_data_org.transpose(1,2,0)       #转换后数据格式为 z  angle x
    # load the elastic parameters
    mat_contents = sio.loadmat(data_dir+'/elastic_VpVsRho_part.mat')   #原数据格式为 x z  参数
    elastic_impedance_data_org = mat_contents['Model']
    elastic_impedance_data = elastic_impedance_data_org.transpose(1,2,0)   #转换后数据格式为 z  参数 x
    elastic_impedance_data = elastic_impedance_data[:,:,1:]        
    # load the initial model
    mat_contents = sio.loadmat(data_dir+'/elastic_VpVsRho_part_initial_smooth400.mat')
    model_initial = mat_contents['Model_initial']
    model_initial = model_initial.transpose(1,2,0)
    model_initial = model_initial[:,:,1:] 

    assert seismic_data.shape[1]==len(args.incident_angles) ,'Data dimensions are not consistent with incident angles. Got {} incident angles and {} in data dimensions'.format(len(args.incident_angles),seismic_data.shape[1])

    seismic_mean = torch.tensor(np.mean(seismic_data,axis=(0,-1),keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data,axis=(0,-1),keepdims=True)).float()
    elastic_mean= torch.tensor(np.mean(elastic_impedance_data,axis=(0,-1), keepdims=True)).float()
    elastic_std = torch.tensor(np.std(elastic_impedance_data,axis=(0,-1),keepdims=True)).float()
    elastic_delt_mean= torch.tensor(np.mean((elastic_impedance_data-model_initial),axis=(0,-1),keepdims=True)).float()
    elastic_delt_std = torch.tensor(np.std((elastic_impedance_data-model_initial),axis=(0,-1),keepdims=True)).float()


    seismic_data = torch.tensor(seismic_data).float()
    elastic_impedance_data = torch.tensor(elastic_impedance_data).float()
    model_initial = torch.tensor(model_initial).float()
    
    if torch.cuda.is_available():
        seismic_data = seismic_data.cuda()
        elastic_impedance_data = elastic_impedance_data.cuda()
        model_initial = model_initial.cuda()
        seismic_mean = seismic_mean.cuda()
        seismic_std = seismic_std.cuda()
        elastic_mean = elastic_mean.cuda()
        elastic_std = elastic_std.cuda()
        elastic_delt_mean = elastic_delt_mean.cuda()
        elastic_delt_std = elastic_delt_std.cuda()

    seismic_normalization = Normalization(mean_val=seismic_mean,
                                          std_val=seismic_std)

    elastic_normalization = Normalization(mean_val=elastic_mean,
                                          std_val=elastic_std)
    elastic_delt_normalization = Normalization(mean_val=elastic_delt_mean,
                                          std_val=elastic_delt_std)

    seismic_data = seismic_normalization.normalize(seismic_data)
    elastic_impedance_data = elastic_normalization.normalize(elastic_impedance_data)

    if not test:
        num_samples = seismic_data.shape[0]
        indecies = np.arange(0,num_samples)
        # 加载训练井数据位置
        mat_contents = sio.loadmat(data_dir+'/train_well_position.mat')
        train_indecies = mat_contents['train_well_position'][0]        
        print(str(train_indecies))
        sio.savemat(args.save_dir+'/train_well_position.mat', {'train_well_position': train_indecies})

        # data.TensorDataset和data.DataLoader用来包装自己的数据，进行批训练
        train_data = data.Subset(data.TensorDataset(seismic_data,elastic_impedance_data,model_initial), train_indecies)   #地震数据 弹性阻抗数据  以及 初始模型 之后抽样
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        unlabeled_loader = data.DataLoader(data.TensorDataset(seismic_data,model_initial), batch_size=args.batch_size, shuffle=True)
        return train_loader, unlabeled_loader, seismic_normalization, elastic_normalization, elastic_delt_normalization
    else:
        test_loader = data.DataLoader(data.TensorDataset(seismic_data,elastic_impedance_data,model_initial), batch_size=args.batch_size, shuffle=False, drop_last=False)
        return test_loader, seismic_normalization, elastic_normalization, elastic_delt_normalization

def get_models(args):

    if args.test_checkpoint is None:
        inverse_net = inverse_model(in_channels=len(args.incident_angles), out_channels=len(args.out_params), nonlinearity=args.nonlinearity)
    else:
        try:
            inverse_net = torch.load(args.test_checkpoint)
        except FileNotFoundError:
            print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
            exit()

    #Set up forward model
    # For simpicity, the same wavlet is used for all incident angles
    wavelet = wavelets.ricker(0.2, 1e-3, 35) # duration of time, time sampling interval and main frequency
    # plt.figure()
    # plt.plot(wavelet)
    wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
    forward_net = forward_model_zoepp(wavelet=wavelet,incident_angles=args.incident_angles,resolution_ratio=args.resolution_ratio)

    if torch.cuda.is_available():
        inverse_net.cuda()
        forward_net.cuda()

    return inverse_net, forward_net

def train(args):

    train_loader, unlabeled_loader, seismic_normalization, elastic_normalization, elastic_delt_normalization = get_data(args)
    inverse_net, forward_net = get_models(args)
    inverse_net.train()
    criterion = nn.MSELoss()
    optimizer = inverse_net.optimizer

    #make a direcroty to save models if it doesn't exist
    if not isdir(args.save_dir+"/checkpoints"):
        os.mkdir(args.save_dir+"/checkpoints")

    print("Training the model")
    best_loss = np.inf
    train_loss = []
    train_property_corr = []
    train_property_r2 = []
    for epoch in tqdm(range(args.max_epoch)):
        for x,y,y_initial in train_loader:
            optimizer.zero_grad()

            y_pred_delt = inverse_net(x)
            y_pred_delt = elastic_delt_normalization.unnormalize(y_pred_delt)
            y_pred = y_initial + y_pred_delt
            y_pred = elastic_normalization.normalize(y_pred)
            # print("y_pred.shape = {}".format(y_pred.size()))
            # print("y.shape = {}".format(y.size()))
            property_loss = criterion(y_pred,y)
            corr, r2 = metrics(y_pred.detach(),y.detach())
            train_property_corr.append(corr)
            train_property_r2.append(r2)

            if args.beta!=0:
                #loading unlabeled data
                try:
                    temp = next(unlabeled)
                    x_u = temp[0]
                    y_u_initial = temp[1]
                except:
                    unlabeled = iter(unlabeled_loader)
                    temp = next(unlabeled)
                    x_u = temp[0]
                    y_u_initial = temp[1]

                y_u_pred_delt = inverse_net(x_u)
                y_u_pred_delt = elastic_delt_normalization.unnormalize(y_u_pred_delt)
                y_u_pred = y_u_initial + y_u_pred_delt
                # print("y_u_pred.shape = {}".format(y_u_pred.size()))
                x_u_rec = forward_net(y_u_pred)
                # print("x_u_rec.shape = {}".format(x_u_rec.size()))
                x_u_rec = seismic_normalization.normalize(x_u_rec)                
                # print("x_u.shape = {}".format(x_u.size()))
                
                seismic_loss = criterion(x_u_rec,x_u)
            else:
                seismic_loss=0

            loss = args.alpha*property_loss + args.beta*seismic_loss
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.detach().clone())
            print(" property_loss = {} ".format(property_loss.detach().clone()))
            # print(" seismic_loss = {} ".format(seismic_loss.detach().clone()))
            print(" loss = {}".format(loss.detach().clone()))            

    # list类型转化为tensor类型
    # print(type(train_loss))
    train_loss = torch.tensor(train_loss)
    # print(train_loss.type())
    if torch.cuda.is_available():
        train_loss = train_loss.cpu()
    # tensor类型转化为numpy类型
    train_loss = train_loss.numpy()
    # print(type(train_loss))    
    
    torch.save(inverse_net,args.save_dir+"/checkpoints/{}".format(args.session_name))
    sio.savemat(args.save_dir+'/train_loss.mat', {'train_loss': train_loss})       

def test(args):
    #make a direcroty to save precited sections
    if not isdir(args.save_dir+"/output_images"):
        os.mkdir(args.save_dir+"/output_images")

    test_loader, seismic_normalization, elastic_normalization, elastic_delt_normalization = get_data(args, test=True)
    if args.test_checkpoint is None:
        args.test_checkpoint = args.save_dir+"./checkpoints/{}".format(args.session_name)
    inverse_net, forward_net = get_models(args)
    criterion = nn.MSELoss(reduction="sum")
    predicted_impedance = []
    true_impedance = []
    test_property_corr = []
    test_property_r2 = []
    inverse_net.eval()
    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for x,y,y_initial in test_loader:
            y_pred_delt = inverse_net(x)
            y_pred_delt = elastic_delt_normalization.unnormalize(y_pred_delt)
            y_pred = y_initial + y_pred_delt
            y_pred = elastic_normalization.normalize(y_pred)                        
            property_loss = criterion(y_pred,y)/np.prod(y.shape)
            corr, r2 = metrics(y_pred.detach(),y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            x_rec = forward_net(elastic_normalization.unnormalize(y_pred))
            x_rec = seismic_normalization.normalize(x_rec)
            seismic_loss = criterion(x_rec, x)/np.prod(x.shape)
            loss = args.alpha*property_loss + args.beta*seismic_loss
            test_loss.append(loss.item())

            true_impedance.append(y)
            predicted_impedance.append(y_pred)

        display_results_Zoepp(test_loss, test_property_corr, test_property_r2, args, header="Test")

        # print(type(predicted_impedance))
        # list类型转化为了tensor类型
        predicted_impedance = torch.cat(predicted_impedance, dim=0) # 将张量按照维度0拼接在一起
        true_impedance = torch.cat(true_impedance, dim=0)
        # print(predicted_impedance.type())

        predicted_impedance = elastic_normalization.unnormalize(predicted_impedance)
        true_impedance = elastic_normalization.unnormalize(true_impedance)

        if torch.cuda.is_available():
            predicted_impedance = predicted_impedance.cpu()
            true_impedance = true_impedance.cpu()
        
        # tensor类型转化为numpy类型
        predicted_impedance = predicted_impedance.numpy()
        true_impedance = true_impedance.numpy()
        # print(type(predicted_impedance))
        
        # save the result
        sio.savemat(args.save_dir+'/predicted_values.mat', {'predicted_values': predicted_impedance})        
        sio.savemat(args.save_dir+'/true_values.mat', {'true_values': true_impedance})
                              
        #diplaying estimated section
        cols = ['{}'.format(col) for col in ['Predicted values','True values', 'Absolute difference']]
        rows = [r'{}'.format(row) for row in args.out_params]
        fig, axes = plt.subplots(nrows=len(args.out_params), ncols=3)

        for i, theta in enumerate(args.out_params):
            axes[i][0].imshow(predicted_impedance[:,i].T, cmap='rainbow',aspect=0.5, vmin=true_impedance.min(), vmax=true_impedance.max())
            axes[i][0].axis('off')
            axes[i][1].imshow(true_impedance[:,i].T, cmap='rainbow',aspect=0.5,vmin=true_impedance.min(), vmax=true_impedance.max())
            axes[i][1].axis('off')
            axes[i][2].imshow(abs(true_impedance[:,i].T-predicted_impedance[:,i].T), cmap='gray',aspect=0.5)
            axes[i][2].axis('off')

        pad = 10 # in points
        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row,xy=(0,0.5), xytext=(-pad,0), xycoords='axes fraction', textcoords='offset points', ha='right', va='center')
        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')

        fig.tight_layout()
        plt.savefig(args.save_dir+"/output_images/{}.png".format(args.test_checkpoint.split("/")[-1]))

        plt.show()



if __name__ == '__main__':
    ## Arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_train_wells', type=int, default=10, help="Number of EI traces from the model to be used for validation")
    parser.add_argument('-max_epoch', type=int, default=500, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=40,help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None,help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'),help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh",help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])
    
    ## Do not change these values unless you use the code on a different data and edit the code accordingly 
    parser.add_argument('-dt', type=float, default=1e-3, help='Time resolution in seconds')
    parser.add_argument('-wavelet_duration',  type=float, default=0.2, help='wavelet duration in seconds')
    # parser.add_argument('-f', default="5, 10, 60, 80", help="Frequency of wavelet. if multiple frequencies use , to seperate them with no spaces, e.g., -f \"5,10,60,80\"", type=lambda x: np.squeeze(np.array(x.split(",")).astype(float)))
    parser.add_argument('-f', type=float, default=35, help="Frequency of Ricker wavelet.")
    parser.add_argument('-resolution_ratio', type=int, default=6, action="store",help="resolution mismtach between seismic and EI")
    parser.add_argument('-incident_angles', type=float, default=np.arange(0, 30+ 1, 2), help="Incident angles of the input seismic and EI")
    # parser.add_argument('-incident_angles', type=float, default=np.arange(0, 30+ 1, 10), help="Incident angles of the input seismic and EI")
    parser.add_argument('-out_params', type=str, default=["vp","vs","density"], help="Output parameters of the network")
 
    args = parser.parse_args()

    
    # set the parameters
    args.num_train_wells = 5
    # args.wavelet_duration = 0.8
    # args.f = 15
    args.max_epoch = 100
    args.resolution_ratio = 6
    args.beta = 0.2
    
    args.save_dir = "./result/Marmousi2/SSL_initial"
    if not isdir(args.save_dir):
        os.mkdir(args.save_dir)   
 
    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)
