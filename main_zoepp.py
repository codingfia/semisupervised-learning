# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:31:13 2021

@author: Yingying Wang

modified from the work of "Alfarraj M, AlRegib G. Semisupervised sequence modeling for elastic impedance inversion[J]. Interpretation, 2019, 7(3): SE237-SE249."

for Marmousi data
"""

import torch
from torch.nn.functional import conv1d
from torch import nn, optim
import numpy as np

class inverse_model(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity="tanh", lr=0.005, lr_decray=1e-4):
        super(inverse_model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.lr = lr
        self.lr_decray = lr_decray
        
        # CNN layers with different dilation rates
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=16, kernel_size=5, padding=2, dilation=1),
                                  nn.GroupNorm(num_groups=8, num_channels=16))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=16, kernel_size=5, padding=6, dilation=3),
                                  nn.GroupNorm(num_groups=8, num_channels=16))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=16, kernel_size=5, padding=12, dilation=6),
                                  nn.GroupNorm(num_groups=8, num_channels=16))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=48, out_channels=32, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=32),
                                 self.activation,
                                 nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=32),
                                 self.activation,
                                 nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
                                 nn.GroupNorm(num_groups=8, num_channels=32),
                                 self.activation)

        self.gru = nn.GRU(input_size=self.in_channels, hidden_size=16, num_layers=3, batch_first=True, bidirectional=True)

        self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=32, out_channels=16, stride=3, kernel_size=5, padding=1),
                                nn.GroupNorm(num_groups=8, num_channels=16),
                                self.activation,
                                nn.ConvTranspose1d(in_channels=16, out_channels=16, stride=2, kernel_size=4, padding=1),
                                nn.GroupNorm(num_groups=8, num_channels=16),
                                self.activation)

        self.gru_out = nn.GRU(input_size=16, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(in_features=32, out_features=self.out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), self.lr, weight_decay=self.lr_decray)
        print("lr = {}".format(self.lr))
        
    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))
        
        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)
        
        x = rnn_out + cnn_out
        
        x = self.up(x)  # for Marmousi data
        
        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)

        x = self.out(x)
        x = x.transpose(-1, -2)
        
        return x

class forward_model_zoepp(nn.Module):
    def __init__(self, wavelet, incident_angles, resolution_ratio=6):
        super(forward_model_zoepp, self).__init__()
        self.wavelet = wavelet.float() if torch.is_tensor(wavelet) else torch.tensor(wavelet).float()
        self.resolution_ratio = resolution_ratio  # vertical scale mismatch between seismic and EI
        self.incident_angles = incident_angles

    def cuda(self):
        self.wavelet = self.wavelet.cuda()

    def zoeppritz_rpp(self, vp1, vs1, rho1, vp2, vs2, rho2, theta1):
        """
        Exact Zoeppritz from expression.
        This is useful because we can pass arrays to it, which we can't do to scattering_matrix().
        Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
        Returns the complex reflectivity.
        Args:
            vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
            vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
            rho1 (ndarray): The upper layer's density; float or 1D array length m.
            vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
            vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
            rho2 (ndarray): The lower layer's density; float or 1D array length m.
            theta1 (ndarray): The incidence angle; float or 1D array length n.
        Returns:
            ndarray. The exact Zoeppritz solution for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
        """
                
        theta1 = theta1 * np.pi / 180
        p = torch.sin(theta1) / vp1  # Ray parameter
        theta2 = torch.asin(p * vp2)
        phi1 = torch.asin(p * vs1)  # Reflected S
        phi2 = torch.asin(p * vs2)  # Transmitted S
    
        a = rho2 * (1 - 2 * torch.sin(phi2)**2.) - rho1 * (1 - 2 * torch.sin(phi1)**2.)
        b = rho2 * (1 - 2 * torch.sin(phi2)**2.) + 2 * rho1 * torch.sin(phi1)**2.
        c = rho1 * (1 - 2 * torch.sin(phi1)**2.) + 2 * rho2 * torch.sin(phi2)**2.
        d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)
    
        E = (b * torch.cos(theta1) / vp1) + (c * torch.cos(theta2) / vp2)
        F = (b * torch.cos(phi1) / vs1) + (c * torch.cos(phi2) / vs2)
        G = a - d * torch.cos(theta1)/vp1 * torch.cos(phi2)/vs2
        H = a - d * torch.cos(theta2)/vp2 * torch.cos(phi1)/vs1
    
        D = E * F + G * H * p**2
        rpp = (1 / D) * (F * (b * (torch.cos(theta1) / vp1) - c * (torch.cos(theta2) / vp2)) - H * p**2 * (a + d * (torch.cos(theta1) / vp1) * (torch.cos(phi2) / vs2)))

        return rpp

    def forward(self, x):
        vp = x[:, 0, :]
        vs = x[:, 1, :]
        rho = x[:, 2, :]        
        vp1, vp2 = vp[:, :-1], vp[:, 1:]
        vs1, vs2 = vs[:, :-1], vs[:, 1:]
        rho1, rho2 = rho[:, :-1], rho[:, 1:]                
        incident_angles = torch.from_numpy(self.incident_angles)
           
        for i in range(incident_angles.size()[0]):
            rc_temp = self.zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, incident_angles[i])
            rc_temp = torch.unsqueeze(rc_temp, 1)
            if i == 0:
                rc = rc_temp
            else:
                rc = torch.cat((rc, rc_temp), 1) 

        for i in range(rc.shape[1]):
            tmp_synth = conv1d(rc[:, [i]], self.wavelet, padding=int(self.wavelet.shape[-1] / 2))
            if i == 0:
                synth = tmp_synth
            else:
                synth = torch.cat((synth, tmp_synth), dim=1)       

        synth = synth[..., ::self.resolution_ratio]  # downsampling
        
        return synth
