##tetracene層内計算
import os
os.environ['HOME'] ='/home/ohno'
import pandas as pd
import time
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.environ['HOME'],'Working/interaction/'))
from make_plane_xyz import exec_gjf##計算した点のxyzfileを出す
from vdw_8_xyz import vdw_R##同様
from utils import get_E
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random

def main_process(args):
    auto_dir = args.auto_dir
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['x1','y1','z1','x2','y2','z2','E','machine_type','status','file_name'])##いじる
        df_E.to_csv(auto_csv_path,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args.auto_dir,args.monomer_name,args.num_nodes,args.max_nodes,args.isTest)##argsの中身を取る
        time.sleep(1)

def listen(auto_dir,monomer_name,num_nodes,max_nodes,isTest):##args自体を引数に取るか中身をばらして取るかの違い
    maxnum_machine2 = 3#int(num_nodes/2) ##多分俺のために空けていてくださったので2 3にする
    
    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
    machine_type_list = df_queue['machine_type'].values.tolist()
    len_queue = len(df_queue)
    
    fixed_param_keys = ['x1','y1','z1'];opt_param_keys = ['x2','y2','z2']
        
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
    
    df_qw = df_E[df_E['status'] == 'qw']
    df_inprogress = df_E[df_E['status'] == 'InProgress']
    len_queue = len(df_inprogress)
    len_qw = len(df_qw)
    margin = max_nodes - len_queue

    if len_qw > 0 and margin > 0:
        # 進行中ジョブのマシンタイプをカウント
        machine_counts = df_inprogress['machine_type'].value_counts().to_dict()
        machine_counts.setdefault(1, 0)
        machine_counts.setdefault(2, 0)
        
        for index, row in df_qw.iterrows():
            if margin == 0:
                break
            params_dict = row[fixed_param_keys + opt_param_keys].to_dict()# パラメータの辞書を作成
            machine_type = 1 if machine_counts.get(2, 0) >= maxnum_machine2 else 2# マシンタイプの決定
            machine_counts[machine_type] += 1

            file_name = ''
            file_name += monomer_name
            for key,val in params_dict.items():
                if key in ['x1','y1','z1','x2','y2','z2']:
                    val = np.round(val,2)
                elif key in ['A1','A2']:#,'theta']:
                    val = int(val)
                file_name += '_{}={}'.format(key,val)
            file_name += '.log'
            file_name0 = exec_gjf(auto_dir, monomer_name, {**params_dict}, machine_type, isTest=isTest)# ジョブの実行
            # 新しい行を作成
            df_E.at[index, 'machine_type'] = machine_type
            df_E.at[index, 'status'] = 'InProgress'
            df_E.at[index, 'file_name'] = file_name
            margin -= 1
        
        df_E.to_csv(auto_csv, index=False)# データフレームをCSVに保存
    
    dict_matrix = get_params_dict(auto_dir,num_nodes)##更新分を流す
    if len(dict_matrix)!=0:#終わりがまだ見えないなら
        for i in range(len(dict_matrix)):
            params_dict=dict_matrix[i]
            print(params_dict)
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            
            df_queue = df_E.loc[df_E['status']=='InProgress',['machine_type','file_name']]
            len_queue = len(df_queue)
            
            isAvailable = len_queue < max_nodes 
            if isAvailable:
                machine_type_list = df_queue['machine_type'].values.tolist()
                machine2IsFull = machine_type_list.count(2) >= maxnum_machine2
                machine_type = 1 if machine2IsFull else 2
                if not(alreadyCalculated):
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict}, machine_type,isTest=isTest)
                    df_newline = pd.Series({**params_dict,'E':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                    df_E=df_E.append(df_newline,ignore_index=True)
                    df_E.to_csv(auto_csv,index=False, float_format='%.2f')
            else:
                if not(alreadyCalculated):
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict}, 1,isTest=True)
                    df_newline = pd.Series({**params_dict,'E':0.,'machine_type':1,'status':'qw','file_name':file_name})
                    df_E=df_E.append(df_newline,ignore_index=True)
                    df_E.to_csv(auto_csv,index=False, float_format='%.2f')
    
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
    fixed_param_keys = ['x1','y1','z1']
    opt_param_keys = ['x2','y2','z2']

    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        #print(1)
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
            return [params_dict]
    dict_matrix=[]
    for index in df_init_params_inprogress.index:##こちら側はinit_params内のある業に関する探索が終わった際の新しい行での探索を開始するもの ###ここを改良すればよさそう
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)
        if isDone:
            opt_params_dict={'x2':np.round(opt_params_matrix[0][0],1),'y2':np.round(opt_params_matrix[0][1],1),'z2':np.round(opt_params_matrix[0][2],1)}
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:##もうこれ以上は新しい計算は進まない
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':##計算が始まっていないものがあったらこの時点で開始する　ここでダメでもまた直にlistenでgrt_params_dictまでいけば新しいのが始まる            
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                dict_matrix.append({**fixed_params_dict,**opt_params_dict})
            else:
                continue

        else:
            for i in range(len(opt_params_matrix)):
                opt_params_dict={'x2':np.round(opt_params_matrix[i][0],1),'y2':np.round(opt_params_matrix[i][1],1),'z2':np.round(opt_params_matrix[i][2],1)}
                df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
                df_qw = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'qw'})
                if (len(df_inprogress)>=1) or (len(df_qw)>=1):
                    continue
                else:
                    d={**fixed_params_dict,**opt_params_dict}
                    dict_matrix.append(d)
                    #print(d)
    return dict_matrix
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    x1 = init_params_dict['x1'];y1 = init_params_dict['y1']; z1 = init_params_dict['z1']
    x2_init_prev = init_params_dict['x2'];y2_init_prev = init_params_dict['y2']; z2_init_prev = init_params_dict['z2']
    while True:
        E_list=[];xyz2_list=[]
        para_list=[]
        for x2 in [x2_init_prev]:
            for y2 in [y2_init_prev-0.1,y2_init_prev,y2_init_prev+0.1]:
                for z2 in [z2_init_prev-0.1,z2_init_prev,z2_init_prev+0.1]:
                    x2 = np.round(x2,1);y2 = np.round(y2,1);z2 = np.round(z2,1)
                    df_val_xyz = df_val[(df_val['x1']==x1)&(df_val['y1']==y1)&(df_val['z1']==z1)&
                                        (df_val['x2']==x2)&(df_val['y2']==y2)&(df_val['z2']==z2)&
                                        (df_val['status']=='Done')]
                    if len(df_val_xyz)==0:
                        para_list.append([x2,y2,z2])
                        continue
                    xyz2_list.append([x2,y2,z2]);E_list.append(df_val_xyz['E'].values[0])
        if len(para_list) != 0:
            return False,para_list
        x2_init,y2_init,z2_init = xyz2_list[np.argmin(np.array(E_list))]
        if x2_init==x2_init_prev and y2_init==y2_init_prev and z2_init==z2_init_prev:
            return True,[[x2_init,y2_init,z2_init]]
        else:
            x2_init_prev=x2_init;y2_init_prev=y2_init;z2_init_prev=z2_init

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
    
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    parser.add_argument('--max-nodes',type=int,help='max nodes')
    ##maxnum-machine2 がない
    args = parser.parse_args()

    print("----main process----")
    main_process(args)
    print("----finish process----")
    