import numpy as np
from sklearn.decomposition import PCA

def get_E(path_file):
    with open(path_file,'r') as f:
        lines=f.readlines()
    lines_E=[]
    for line in lines:
        if line.find('E(R')>-1 and len(line.split())>5:
            lines_E.append(float(line.split()[4])*627.510)
    E_list=[lines_E[5*i]-lines_E[5*i+1]-lines_E[5*i+2] for i in range(int(len(lines_E)/5))]
    return E_list

def get_E_mono_1(path_file):
    with open(path_file,'r') as f:
        lines=f.readlines()
    lines_E=[]
    for line in lines:
        if line.find('E(R')>-1 and len(line.split())>5:
            lines_E.append(float(line.split()[4])*627.510)
    E = lines_E[-2]+1058790
    return E

def get_E_mono_2(path_file):
    with open(path_file,'r') as f:
        lines=f.readlines()
    lines_E=[]
    for line in lines:
        if line.find('E(R')>-1 and len(line.split())>5:
            lines_E.append(float(line.split()[4])*627.510)
    E = lines_E[-1]+1058790
    return E

def get_E_len(path_file):
    with open(path_file,'r') as f:
        lines=f.readlines()
    lines_E=[]
    for line in lines:
        if line.find('E(R')>-1 and len(line.split())>5:
            lines_E.append(float(line.split()[4])*627.510)
    return len(lines_E)

def squeeze_min_df_E(df_E,columns=['A1','A2']):
    ddf = df_E.groupby(columns)
    df_Emin = df_E.loc[ddf['E'].idxmin(),:]
    return df_Emin

def get_rot_axis_from_A2(A2,glide_mode):
    A2 = np.radians(A2)
    if glide_mode=='a':
        rot_axis_i = np.array([-np.sin(A2),np.cos(A2),0.])
        rot_axis_t = np.array([-np.sin(-A2),np.cos(-A2),0.])
    else:
        rot_axis_i = np.array([-np.sin(A2),np.cos(A2),0.])
        rot_axis_t = np.array([-np.sin(np.pi-A2),np.cos(np.pi-A2),0.])
    
    return rot_axis_i, rot_axis_t

#Ra,Rb,heri/2 --> R1,R2
def convertor_R(Ra,Rb,theta_):
    R1=Ra*np.cos(theta_)+Rb*np.sin(theta_)
    R2=-Ra*np.sin(theta_)+Rb*np.cos(theta_)
    return R1,R2

# nの周りにtheta_in回転する回転行列
def Rod(n,theta_in):
    nx,ny,nz=n
    theta_t=np.radians(theta_in)
    Rod=np.array([[np.cos(theta_t)+(nx**2)*(1-np.cos(theta_t)),nx*ny*(1-np.cos(theta_t))-nz*np.sin(theta_t),nx*nz*(1-np.cos(theta_t))+ny*np.sin(theta_t)],
                [nx*ny*(1-np.cos(theta_t))+nz*np.sin(theta_t),np.cos(theta_t)+(ny**2)*(1-np.cos(theta_t)),ny*nz*(1-np.cos(theta_t))-nx*np.sin(theta_t)],
                [nx*nz*(1-np.cos(theta_t))-ny*np.sin(theta_t),ny*nz*(1-np.cos(theta_t))+nx*np.sin(theta_t),np.cos(theta_t)+(nz**2)*(1-np.cos(theta_t))]])
    return Rod

def extract_axis(xyz_array):#shape=[n,3]
    pca = PCA()
    pca.fit(xyz_array)
    long_axis = pca.components_[0]
    short_axis = pca.components_[1]
    return long_axis, short_axis

def heri_to_A3(A1,A2,heri):
    N=361
    A1=np.radians(A1);A2=np.radians(A2)
    ax1=np.array([np.sin(A1)*np.cos(A2),np.sin(A1)*np.sin(A2),np.cos(A1)])
    ax2=np.array([np.sin(A1)*np.cos(A2),-np.sin(A1)*np.sin(A2),np.cos(A1)])
    heri_list=np.zeros(N,dtype='float64');error_list=np.zeros(N,dtype='float64')
    A3_list=np.array([round(A3) for A3 in np.linspace(-180,180,N)])
    n1 = np.array([-np.cos(A1)*np.cos(A2),-np.cos(A1)*np.sin(A2),np.sin(A1)])
    n2 = np.array([-np.cos(A1)*np.cos(A2),+np.cos(A1)*np.sin(A2),np.sin(A1)])
    for i,A3 in enumerate(A3_list):
        ex1=np.matmul(Rod(ax1,A3),n1)
        ex2=np.matmul(Rod(ax2,-A3),n2)
        ex21_cross = np.cross(ex2,ex1)
        exS=np.matmul(Rod(ax1,A3-90),n1)
        
        isDirectedToB = exS[1]>0
        isOpenHB = ex21_cross[2]>0
        heri_abs = np.degrees(np.arccos(np.dot(ex1,ex2)))
        if isOpenHB & isDirectedToB:#どちらの八の字か？上向いてるか?
            heri_list[i] = heri_abs
        else:
            heri_list[i] = float('inf')#計算めんどいので例外にする
        error_list[i]=abs(heri_list[i]-heri)
    idx=np.argsort(error_list);heri_sort=heri_list[idx];A3_sort=A3_list[idx]
    A3_1=A3_sort[0]
    return A3_1

def R2atom(R):
    if R==1.8:
        return 'S'
    elif R==1.7:
        return 'C'
    elif R==1.2:
        return 'H'
    else:
        return 'X'

def get_ab_from_params(R1,R2,heri):
    A_rad=np.radians(heri/2)
    a_=2*(R1*np.cos(A_rad)-R2*np.sin(A_rad))
    b_=2*(R2*np.cos(A_rad)+R1*np.sin(A_rad))
    return a_, b_ 

def getA1_from_R3t(a,R3t,glide):
    assert glide=='a'
    return np.rad2deg(np.arctan(R3t/(a/2)))

def check_calc_status(df_cur,A1,A2,A3,a,b):
    try:        
        return df_cur.loc[
                        (df_cur['A1']==A1)&
                        (df_cur['A2']==A2)&
                        (df_cur['A3']==A3)&
                        (df_cur['a']==a)&
                        (df_cur['b']==b), 'status'].values[0] == 'Done'
    except IndexError:
        return False

def convert_A_df(df):
    A1_array = df['A1'].values
    A2_array = df['A2'].values
    df['A1_new'] = np.degrees(np.arcsin(np.sin(np.radians(A1_array))*np.cos(np.radians(A2_array))))
    df['A2_new'] = np.degrees(np.arctan(np.tan(np.radians(A1_array))*np.sin(np.radians(A2_array))))
    return df

def convert_A(A1,A2):
    A1_new = np.degrees(np.arcsin(np.sin(np.radians(A1))*np.cos(np.radians(A2))))
    A2_new = np.degrees(np.arctan(np.tan(np.radians(A1))*np.sin(np.radians(A2))))
    return A1_new, A2_new

def invert_A(A1,A2):
    A1_old = np.degrees(np.arccos(np.cos(np.radians(A1))*np.cos(np.radians(A2))))
    if A1==0:
        A2_old = 90 if A2>0 else -90
    else:
        A2_old = np.degrees(np.arctan(np.sin(np.radians(A2))/np.tan(np.radians(A1))))
    
    def translator_A(_A1_new, _A2_new, _A1_old, _A2_old):
        if _A1_new>=0:
            return _A1_old, _A2_old
        elif _A2_new>0:
            return _A1_old, _A2_old+180.0
        elif _A2_new==0:
            return _A1_old, _A2_old
        elif _A2_new<0:
            return _A1_old, _A2_old-180.0
    
    A1_old, A2_old = translator_A(A1,A2,A1_old, A2_old)
    
    return A1_old, A2_old

def phi_into_180(phi):
    if phi>180:
        return phi - 360
    elif phi<-180:
        return phi + 360
    else:
        return phi
