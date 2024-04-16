##tetracene層内計算
import os
os.environ['HOME'] ='/home/ohno'
import pandas as pd
import time
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.environ['HOME'],'Working/interaction/'))
from make_8_xyz import exec_gjf##計算した点のxyzfileを出す
from vdw_8_xyz import vdw_R##同様
from utils import get_E
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random

def init_process(args):
    # 数理モデル的に自然な定義の元のparams initリスト: not yet
    # 結晶学的に自然なパラメータへ変換: not yet
    auto_dir = args.auto_dir
    monomer_name = args.monomer_name
    order = 5
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)

    def get_init_para_csv(auto_dir,monomer_name):
        init_params_csv = os.path.join(auto_dir, 'step1_init_params.csv')
        
        init_para_list = []
        A1 = 0; A2 = 0
        theta_list = [5,10,15,20,25,30,35,40,45]#list(range(0,95,5)) ##+ [22.5,23.0,23.5,24.0,24.5] + [25.5,26.0,26.5,27.0,27.5]##全体を掃く＋細かいところ
        for theta in tqdm(theta_list):
            a_list = []; b_list = []; S_list = []
            a_clps=vdw_R(A1,A2,theta,0.0,'a',monomer_name)
            b_clps=vdw_R(A1,A2,theta,90.0,'b',monomer_name)
            for theta_ab in range(0,91):
                R_clps=vdw_R(A1,A2,theta,theta_ab,'t',monomer_name)
                a=2*R_clps*np.cos(np.radians(theta_ab))
                b=2*R_clps*np.sin(np.radians(theta_ab))
                if (a_clps > a) or (b_clps > b):
                    continue
                else:
                    a1 = np.round(a,1);b1 = np.round(b,1)
                    a_list.append(a1);b_list.append(b1);S_list.append(a*b)##Sを丸めずに出力
            local_minidx_list = signal.argrelmin(np.array(S_list), order=order)
            if len(local_minidx_list[0])>0:
                for local_minidx in local_minidx_list[0]:
                    init_para_list.append([a_list[local_minidx],b_list[local_minidx],theta,'NotYet'])
            init_para_list.append([a_list[0],b_list[0],theta,'NotYet'])
            init_para_list.append([a_list[-1],b_list[-1],theta,'NotYet'])
            
        df_init_params = pd.DataFrame(np.array(init_para_list),columns = ['a','b','theta','status'])
        df_init_params.to_csv(init_params_csv,index=False)
    
    get_init_para_csv(auto_dir,monomer_name)
    
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E_init = pd.DataFrame(columns = ['a','b','theta','E','E_p1','E_p2','E_t','machine_type','status','file_name'])##隣接8分子
    else:
        df_E_init = pd.read_csv(auto_csv_path)
        df_E_init = df_E_init[df_E_init['status']!='InProgress']
    df_E_init.to_csv(auto_csv_path,index=False)

    df_init=pd.read_csv(os.path.join(auto_dir,'step1_init_params.csv'))
    df_init['status']='NotYet'
    df_init.to_csv(os.path.join(auto_dir,'step1_init_params.csv'),index=False)

def main_process(args):
    auto_dir = args.auto_dir
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['x','y','z','E','machine_type','status','file_name'])##いじる
        df_E.to_csv(auto_csv_path,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args.auto_dir,args.monomer_name,args.num_nodes,args.isTest)##argsの中身を取る
        time.sleep(1)

def listen(auto_dir,monomer_name,num_nodes,isTest):##args自体を引数に取るか中身をばらして取るかの違い
    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
    machine_type_list = df_queue['machine_type'].values.tolist()
    len_queue = len(df_queue)
    maxnum_machine2 = 3#int(num_nodes/2) ##多分俺のために空けていてくださったので2 3にする
    
    for idx,row in zip(df_queue.index,df_queue.values):
        machine_type,file_name = row
        log_filepath = os.path.join(*[auto_dir,'gaussian',file_name])
        if not(os.path.exists(log_filepath)):#logファイルが生成される直前だとまずいので
            continue
        E_list=get_E(log_filepath)
        if len(E_list)!=1:##get Eの長さは計算した分子の数
            continue
        else:
            len_queue-=1;machine_type_list.remove(machine_type)
            E=float(E_list[0])##8分子に向けてep1,ep2作成　ep1:b ep2:a
            df_E.loc[idx, ['E','status']] = [E,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    isAvailable = len_queue < num_nodes 
    machine2IsFull = machine_type_list.count(2) >= maxnum_machine2
    machine_type = 1 if machine2IsFull else 2
    if isAvailable:
        params_dict = get_params_dict(auto_dir,num_nodes)
        if len(params_dict)!=0:#終わりがまだ見えないなら
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                file_name = exec_gjf(auto_dir, monomer_name, {**params_dict}, machine_type,isTest=isTest)##計算を実行並びにxyzファイルの出力
                df_newline = pd.Series({**params_dict,'E':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes):
    """
    前提:
        step1_init_params.csvとstep1.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    fixed_param_keys = ['x','z']
    opt_param_keys = ['y']

    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        print(1)
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return params_dict
    for index in df_init_params.index:
        df_init_params = pd.read_csv(init_params_csv)
        print('index')
        print(index)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        isDone, opt_params_dict = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)
        if isDone:
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':                
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                return {**fixed_params_dict,**opt_params_dict}
            else:
                continue

        else:
            df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
            print(df_inprogress)
            if len(df_inprogress)>=1:
                print('continue')
                continue
            return {**fixed_params_dict,**opt_params_dict}
    return {}
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    x = init_params_dict['x'];y_init_prev = init_params_dict['y']; z = init_params_dict['z']
    while True:
        E_list=[];y_list=[]
        for y in [y_init_prev-0.1,y_init_prev,y_init_prev+0.1]:
            y = np.round(y,1)
            df_val_y = df_val[(df_val['x']==x)&(df_val['y']==y)&(df_val['z']==z)&(df_val['status']=='Done')]
            if len(df_val_y)==0:
                return False,{'y':y}
            y_list.append(y);E_list.append(df_val_y['E'].values[0])
        y_init = y_list[np.argmin(np.array(E_list))]
        if y_init==y_init_prev:
            return True,{'y':y_init}
        else:
            y_init_prev=y_init

def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    for k, v in dict_filter.items():
        if type(v)==str:
            df=df[df[k]==v]
        else:
            df=df[df[k]==v]
    df_filtered=df
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    ##maxnum-machine2 がない
    args = parser.parse_args()

    if args.init:
        print("----initial process----")
        init_process(args)
    
    print("----main process----")
    main_process(args)
    print("----finish process----")
    