import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import Rod
from make_8_xyz import get_monomer_xyzR
from scipy import signal

def get_c_vec_vdw(monomer_name,A1,A2,a_,b_,theta):#,name_csv
    
    i=np.zeros(3); a=np.array([a_,0,0]); b=np.array([0,b_,0]); t1=(a+b)/2;t2=(a-b)/2 
    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A1,A2,theta)
    
    monomer_array_t = get_monomer_xyzR(monomer_name,0.,0.,0.,-A1,A2,-theta)
    
    arr_list=[[i,'p'],[b,'p'],[-b,'p'],[a,'p'],[-a,'p'],[t1,'t'],[-t1,'t'],[t2,'t'],[-t2,'t']]
    Rb_list=[np.round(Rb,1) for Rb in np.linspace(-np.round(b_/2,1),np.round(b_/2,1),int(np.round(2*np.round(b_/2,1)/0.1))+1)]
    z_list=[];V_list=[]
    
    for Rb in Rb_list:
        z_max=0
        for R,arr in arr_list:
            if arr=='t':
                monomer_array1=monomer_array_t
            elif arr=='p':
                monomer_array1=monomer_array_i
            for x1,y1,z1,R1 in monomer_array1:#層内
                x1,y1,z1=np.array([x1,y1,z1])+R
                for x2,y2,z2,R2 in monomer_array_i:#i0
                    y2+=Rb
                    z_sq=(R1+R2)**2-(x1-x2)**2-(y1-y2)**2
                    if z_sq<0:
                        z_clps=0.0
                    else:
                        z_clps=np.sqrt(z_sq)+z1-z2
                    z_max=max(z_max,z_clps)
        z_list.append(z_max)
        V_list.append(a_*b_*z_max)
    
    return np.array([0,Rb_list[np.argmin(V_list)],z_list[np.argmin(V_list)]])
    
# theta=arctan(b/a)
def vdw_R(A1,A2,A3,theta,dimer_mode,monomer_name):##2分子が接しないようにするにはtheta方向にどれだけずらせばいいのか
    monomer_1=get_monomer_xyzR(monomer_name,Ta=0.,Tb=0.,Tc=0.,A1=A1,A2=A2,A3=A3)
    if dimer_mode=='t':
        monomer_2=get_monomer_xyzR(monomer_name,Ta=0.,Tb=0.,Tc=0.,A1=-A1,A2=A2,A3=-A3)#convertor(monomer,A1,-A2,-A3+glide)
    elif dimer_mode=='a' or dimer_mode=='b':
        monomer_2=get_monomer_xyzR(monomer_name,Ta=0.,Tb=0.,Tc=0.,A1=A1,A2=A2,A3=A3)
    R_clps=0
    for x1,y1,z1,rad1 in monomer_1:
        for x2,y2,z2,rad2 in monomer_2:
            eR=np.array([np.cos(np.radians(theta)),np.sin(np.radians(theta)),0.0])
            R_1b=np.dot(eR,np.array([x1,y1,z1]))
            R_2b=np.dot(eR,np.array([x2,y2,z2]))
            R_12=np.array([x2-x1,y2-y1,z2-z1])
            R_12b=np.dot(eR,R_12)
            R_12a=np.linalg.norm(R_12-R_12b*eR)
            if (rad1+rad2)**2-R_12a**2<0:
                continue
            else:
                R_clps=max(R_clps,R_1b-R_2b+np.sqrt((rad1+rad2)**2-R_12a**2))
    return R_clps

# FF
def get_phi12(monomer_name,A1,A2,a_,b_,theta):
    phi_list = np.linspace(-180.0,180.0,73)
    phi12_list = []; FF_6NN_list = []
    
    for phi1 in tqdm(phi_list):
        for phi2 in phi_list:
            FF_6NN = get_FF_6NN(monomer_name,A1,A2,a_,b_,theta,phi1,phi2)
            phi12_list.append([phi1,phi2]); FF_6NN_list.append(FF_6NN)
            
    FF_6NN_argmin = np.argmin(np.array(FF_6NN_list))
    
    phi1,phi2 = phi12_list[FF_6NN_argmin]
    
    return phi1,phi2

def FF_phi_localmins(monomer_name,A1,A2,a_,b_,theta):
    phi_list = np.linspace(-180.0,180.0,73)
    FF_6NN_list = []
    order = 3
    for phi1 in phi_list:
        phi2 = - phi1
        FF_6NN = get_FF_6NN(monomer_name,A1,A2,a_,b_,theta,phi1,phi2)
        FF_6NN_list.append(FF_6NN)
    
    local_minidx_list = signal.argrelmin(np.array(FF_6NN_list), order=order)
    negative_local_minidx_list = []
    for local_minidx in local_minidx_list[0]:
        if FF_6NN_list[local_minidx]<0:
            negative_local_minidx_list.append(local_minidx)
    return FF_6NN_list, phi_list, negative_local_minidx_list

def get_FF_6NN(monomer_name,A1,A2,a_,b_,theta,phi1,phi2):
    
    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A1,A2,theta,phi1,isFF=True)
    
    if a_>b_:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,0,b_,0,A1,A2,theta,phi1,isFF=True)
        monomer_array_p2 = get_monomer_xyzR(monomer_name,0,-b_,0,A1,A2,theta,phi1,isFF=True)

    else:
        monomer_array_p1 = get_monomer_xyzR(monomer_name,a_,0,0,A1,A2,theta,phi1,isFF=True)
        monomer_array_p2 = get_monomer_xyzR(monomer_name,-a_,0,0,A1,A2,theta,phi1,isFF=True)

    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,0,-A1,A2,-theta,phi2,isFF=True)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,0,-A1,A2,-theta,phi2,isFF=True)
    monomer_array_t3 = get_monomer_xyzR(monomer_name,-a_/2,-b_/2,0,-A1,A2,-theta,phi2,isFF=True)
    monomer_array_t4 = get_monomer_xyzR(monomer_name,-a_/2,b_/2,0,-A1,A2,-theta,phi2,isFF=True)
    monomer_array_6NN_list = [
        monomer_array_p1,monomer_array_t1,monomer_array_t2,
        monomer_array_p2,monomer_array_t3,monomer_array_t4,
    ]

    def get_FF(monomer_array_1, monomer_array_2, C1_index):
        e=1.602176634*(10**(-19))
        
        FF=0
        for x1,y1,z1,R1,q1,sig1,eps1 in monomer_array_1[C1_index:]:
            for x2,y2,z2,R2,q2,sig2,eps2 in monomer_array_2[C1_index:]:
                r=np.linalg.norm([x1-x2,y1-y2,z1-z2])
                q=q1*q2*(e**2)
                sig=np.sqrt(sig1*sig2)
                eps=np.sqrt(eps1*eps2)
                FF+=q/r+4*eps*((sig/r)**12-(sig/r)**6)
        return FF
    
    FF_6NN = 0; C1_index =23
    for monomer_array_NN in monomer_array_6NN_list:
       
        FF = get_FF(monomer_array_i, monomer_array_NN, C1_index)
        FF_6NN += FF
        
    return FF_6NN
    
    