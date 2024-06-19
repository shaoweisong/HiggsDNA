import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
# in jsonpog the contents is looped over the first edge of all the second edges, then the second edge of all the second edges, eg. if the first edge is eta(10bins), second edge is pt(5bins). The sf contents are [eta1,pt1],[eta1,pt2],...[eta1,pt5],[eta2,pt1],...[eta10,pt5]
# 定义路径常量
JQ_UL18_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/18_offical.root"
JQ_UL17_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/17_offical.root"
JQ_UL16_postVFP_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/16_postVFP_offical.root"
JQ_UL16_preVFP_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/16_preVFP_offical.root"

offical_UL18_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/egammaEffi.txt_EGM2D_Pho_wp90.root_UL18.root"
offical_UL17_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/egammaEffi.txt_EGM2D_PHO_MVA90_UL17.root"
offical_UL16_postVFP_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/egammaEffi.txt_EGM2D_Pho_MVA90_UL16_postVFP.root"
offical_UL16_preVFP_path = "/afs/cern.ch/user/s/shsong/HiggsDNA/higgs_dna/systematics/data/egammaEffi.txt_EGM2D_Pho_wp90_UL16.root"

def get_custom_sfs_and_systematic_errors(official_path, jq_path):
    """
    从给定路径加载自定义的标量因子数据，并计算其系统误差。

    参数:
        jq_path (str): 自定义标量因子文件路径。

    返回:
        tuple: 包含标量因子数组和系统误差数组的元组。
    """
    # 打开文件并获取数据
    events_jq = uproot.open(jq_path)['EGamma_SF2D']
    events_official = uproot.open(official_path)['EGamma_SF2D']
    
    # 提取标量因子值和统计误差
    sfs = events_jq.values()
    contents=ak.flatten(sfs)
    #把contents转化成list
    sfslist = contents.tolist()
    # 计算自定义SF与官方SF之间的差异（B）
    diff = events_jq.values() - events_official.values()
    height = diff**2
    
    # 计算官方SF的总不确定性（C）
    total_official_uncer = events_official.errors()**2
    total_stat_uncer = events_jq.errors()**2
    
    # 计算总不确定性
    # total_uncer = np.sqrt(total_stat_uncer + height + total_official_uncer).T
    total_uncer = np.sqrt(total_stat_uncer + height + total_official_uncer)
    total_uncer_list = ak.flatten(total_uncer).tolist()
    return sfslist, total_uncer_list

def get_x_y_bins(years):
    """
    返回:
        tuple: 包含x和y轴bin边界的元组。
    """
    # ETA bin
    x_edges = np.array([-2.5, -2, -1.566, -1.444, -0.8, 0, 0.8, 1.444, 1.566, 2, 2.5])
    if years == 'UL17':
        y_edges = np.array([20, 35, 50, 100, 200, 500]) # pt bin
    else:
        y_edges = np.array([20, 35, 50, 80, 120, 500])  # pt bin
    return 
jq_path_example = JQ_UL18_path
official_path_example = offical_UL18_path
sfslist, total_uncerlist = get_custom_sfs_and_systematic_errors(official_path_example, jq_path_example)
sfsuplist = (np.array(sfslist) + np.array(total_uncerlist)).tolist()
sfsdownlist = (np.array(sfslist) - np.array(total_uncerlist)).tolist()
inputlist=["eta","pt"]
if "UL18" in jq_path_example:
    bins1 =  [-float('inf'), -2.0, -1.566, -1.444, -0.8, 0.0, 0.8, 1.444, 1.566, 2.0, float('inf')]
    bins2 = [20.0, 35.0, 50.0, 80.0, 120.0,500.0]
else:
    bins1 =  [-float('inf'), -2.0, -1.566, -1.444, -0.8, 0.0, 0.8, 1.444, 1.566, 2.0, float('inf')]
    bins2 = [20.0, 35.0, 50.0, 100.0, 200.0, 500.0]
data={"key": "modified_photonID_cut",
      "value":{"nodetype": "multibinning",
      "inputs":inputlist,
      "edges": [bins1, bins2],
      "content": sfslist,
      "flow": "error"}}
dataup={"key": "modified_photonID_cut",
        "value":{"nodetype": "multibinning",
        "inputs":inputlist,
        "edges": [bins1, bins2],
        "content": sfsuplist,
        "flow": "error"}}
datadown={"key": "modified_photonID_cut",
        "value":{"nodetype": "multibinning",
        "inputs":inputlist,
        "edges": [bins1, bins2],
        "content": sfsdownlist,
        "flow": "error"}}
            
with open('modified_photonID.json', 'w') as json_file:
    json_file.write('[\n')
    json.dump(data, json_file, indent=4)
    json_file.write(',\n')
    json.dump(dataup, json_file, indent=4)
    json_file.write(',\n')
    json.dump(datadown, json_file, indent=4)
    json_file.write('\n]')
