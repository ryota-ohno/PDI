import os
import numpy as np
import pandas as pd
import subprocess
from utils import Rod, R2atom

MONOMER_LIST = ["PDI"]
############################汎用関数###########################
def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A1,A2,A3):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono=pd.read_csv('~/Working/PDI/monomer/{}.csv'.format(monomer_name))
    atoms_array_xyzR=df_mono[['atom','x','y','z']].values
    
    ex = np.array([1.,0.,0.]); ey = np.array([0.,1.,0.]); ez = np.array([0.,0.,1.])

    xyz_array = atoms_array_xyzR[:,1:]
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)
    xyz_array = np.matmul(xyz_array,Rod(ey,A1).T)
    xyz_array = xyz_array + T_vec
    atom_array = atoms_array_xyzR[:,0].reshape((-1,1))
    
    if monomer_name in MONOMER_LIST:
        return np.concatenate([atom_array,xyz_array],axis=1)
    
    else:
        raise RuntimeError('invalid monomer_name={}'.format(monomer_name))
        
def get_xyzR_lines(xyzR_array,file_description,machine_type):
    if machine_type==1:
        mp_num = 40
    elif machine_type==2:
        mp_num = 52
    lines = [     
        '%mem=15GB\n',
        f'%nproc={mp_num}\n',
        '#P TEST pbepbe/6-311G** EmpiricalDispersion=GD3BJ counterpoise=2\n',
        '\n',
        file_description+'\n',
        '\n',
        '0 1 0 1 0 1\n'
    ]
    mol_len = len(xyzR_array)//2
    atom_index = 0
    mol_index = 0
    for atom,x,y,z in xyzR_array:
        mol_index = atom_index//mol_len + 1
        line = '{}(Fragment={}) {} {} {}\n'.format(atom,mol_index,x,y,z)     
        lines.append(line)
        atom_index += 1
    return lines

# 実行ファイル作成
def get_one_exe(file_name,machine_type):
    file_basename = os.path.splitext(file_name)[0]
    #mkdir
    if machine_type==1:
        gr_num = 1; mp_num = 40
    elif machine_type==2:
        gr_num = 2; mp_num = 52
    cc_list=[
        '#!/bin/sh \n',
         '#$ -S /bin/sh \n',
         '#$ -cwd \n',
         '#$ -V \n',
         '#$ -q gr{}.q \n'.format(gr_num),
         '#$ -pe OpenMP {} \n'.format(mp_num),
         '\n',
         'hostname \n',
         '\n',
         'export g16root=/home/g03 \n',
         'source $g16root/g16/bsd/g16.profile \n',
         '\n',
         'export GAUSS_SCRDIR=/home/scr/$JOB_ID \n',
         'mkdir /home/scr/$JOB_ID \n',
         '\n',
         'g16 < {}.inp > {}.log \n'.format(file_basename,file_basename),
         '\n',
         'rm -rf /home/scr/$JOB_ID \n',
         '\n',
         '\n',
         '#sleep 5 \n'
#          '#sleep 500 \n'
            ]

    return cc_list

######################################## 特化関数 ########################################

##################gaussview##################
def make_xyzfile(monomer_name,params_dict):
    x = params_dict['x']; y = params_dict['y']; z = params_dict['z']
    A1 = params_dict.get('A1',0.0); A2 = params_dict.get('A2',0.0); A3 = params_dict.get('A3',0.0)

    monomer_array_i = get_monomer_xyzR(monomer_name,0.0,0,0,A1,A2,A3)
    
    monomer_array_p1 = get_monomer_xyzR(monomer_name,x,y,z,A1,A2,A3)##1,2がb方向
    
    xyz_list=['400 \n','polyacene9 \n']##4分子のxyzファイルを作成
    monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p1],axis=0)
    
    for x,y,z,R in monomers_array_4:
        atom = R2atom(R)
        line = '{} {} {} {}\n'.format(atom,x,y,z)     
        xyz_list.append(line)
    
    return xyz_list

def make_xyz(monomer_name,params_dict):
    xyzfile_name = ''
    xyzfile_name += monomer_name
    for key,val in params_dict.items():
        if key in ['x','y','z']:
            val = np.round(val,2)
        elif key in ['A1','A2']:#,'theta']:
            val = int(val)
        xyzfile_name += '_{}={}'.format(key,val)
    return xyzfile_name + '.xyz'

def make_gjf_xyz(auto_dir,monomer_name,params_dict,machine_type):
    x = params_dict['x']; y = params_dict['y']; z = params_dict['z']
    A1 = params_dict.get('A1',0.0); A2 = params_dict.get('A2',0.0); A3 = params_dict.get('A3',0.0)
    
    monomer_array_i = get_monomer_xyzR(monomer_name,0.0,0,0,A1,A2,A3)
    monomer_array_2 = get_monomer_xyzR(monomer_name,x,y,z,A1,A2,A3)
    
    dimer_array_t1 = np.concatenate([monomer_array_i,monomer_array_2])
    
    
    file_description = '{}_x={}_y={}_z={}'.format(monomer_name,round(x,1),round(y,1),round(z,1))
    line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1,file_description+'_t1',machine_type)
    
    gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_t1 + ['\n\n\n']
    
    file_name = get_file_name_from_dict(monomer_name,params_dict)
    os.makedirs(os.path.join(auto_dir,'gaussian'),exist_ok=True)
    gij_xyz_path = os.path.join(auto_dir,'gaussian',file_name)
    with open(gij_xyz_path,'w') as f:
        f.writelines(gij_xyz_lines)
    
    return file_name

def get_file_name_from_dict(monomer_name,params_dict):
    file_name = ''
    file_name += monomer_name
    for key,val in params_dict.items():
        if key in ['x','y','z']:
            val = np.round(val,2)
        elif key in ['A1','A2']:#,'theta']:
            val = int(val)
        file_name += '_{}={}'.format(key,val)
    return file_name + '.inp'
    
def exec_gjf(auto_dir, monomer_name, params_dict, machine_type,isTest=True):
    inp_dir = os.path.join(auto_dir,'gaussian')
    xyz_dir = os.path.join(auto_dir,'gaussview')
    print(params_dict)
    
    xyzfile_name = make_xyz(monomer_name, params_dict)
    xyz_path = os.path.join(xyz_dir,xyzfile_name)
    xyz_list = make_xyzfile(monomer_name,params_dict)
    with open(xyz_path,'w') as f:
        f.writelines(xyz_list)
    
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict,machine_type)
    cc_list = get_one_exe(file_name,machine_type)
    sh_filename = os.path.splitext(file_name)[0]+'.r1'
    sh_path = os.path.join(inp_dir,sh_filename)
    with open(sh_path,'w') as f:
        f.writelines(cc_list)
    if not(isTest):
        subprocess.run(['qsub',sh_path])
    log_file_name = os.path.splitext(file_name)[0]+'.log'
    return log_file_name
    
############################################################################################