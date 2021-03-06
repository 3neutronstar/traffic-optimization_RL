import torch
import numpy as np
node_id=["n_nl","n_nr",
"n_ln","n_cnl","n_cnr","n_rn",
"n_ls","n_csl","n_csr","n_rs",
"n_sl" ,"n_sr" ]

edge_dict={
    "n_nl":["n_cnl"],
    "n_cnl":["n_nl","n_ln","n_cnr","n_csl"],
    "n_nr":["n_cnr"],
    "n_cnr":["n_nr","n_cnl","n_rn","n_csr"],
    "n_ln":["n_cnl"],
    "n_rn":["n_cnr"],
    "n_csl":["n_cnl","n_ls","n_csr","n_sl"],
    "n_csr":["n_cnr","n_csl","n_rs","n_sr"],
    "n_ls":["n_csl"],
    "n_rs":["n_csr"],
    "n_sl":["n_csl"],
    "n_sr":["n_csr"]
}
def specify_edgeid(edge_dict):
    num_lanes=3
    edges=list()
    for _, dict_key in enumerate(edge_dict.keys()):
        for i, _ in enumerate(edge_dict[dict_key]):
            print('    <edge from="{}" id="{}_to_{}" to="{}" numLanes="{}"/>'.format(dict_key,
            dict_key,edge_dict[dict_key][i],edge_dict[dict_key][i],num_lanes))
            edges.append("{}_to_{}".format(dict_key,edge_dict[dict_key][i]))
    return edges
    

edge_list=specify_edgeid(edge_dict)    

def specify_flow(edge_dict):
    edge_start=list()
    node_fin=list()
    flows=list()
    edge_fin=list()
    for _,key_start in enumerate(edge_dict):
        if len(edge_dict[key_start])==1:
            append_start_str="{}_to_{}".format(key_start,edge_dict[key_start][0])
            edge_start.append(append_start_str)
            if key_start[2]=='n':
                key_fin=key_start[:2]+'s'+key_start[3:]
                for _,dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i]==key_fin:
                            node_fin=dict_key
                append_fin_str="{}_to_{}".format( node_fin,key_fin)
            elif key_start[2]=='s':
                key_fin=key_start[:2]+'n'+key_start[3:]
                for _,dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i]==key_fin:
                            node_fin=dict_key
                append_fin_str="{}_to_{}".format( node_fin,key_fin)  
            elif key_start[2]=='r':
                key_fin=key_start[:2]+'l'+key_start[3:]
                for _,dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i]==key_fin:
                            node_fin=dict_key
                append_fin_str="{}_to_{}".format( node_fin,key_fin)
            elif key_start[2]=='l':
                key_fin=key_start[:2]+'r'+key_start[3:]
                for _,dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i]==key_fin:
                            node_fin=dict_key
                append_fin_str="{}_to_{}".format( node_fin,key_fin)
            edge_fin.append(append_fin_str)
            flows.append('from="{}" to="{}"'.format(append_start_str,append_fin_str))
            print('    <flow id="{}" from="{}" to="{}" begin="0" end="9000" number="1800" />'.format(key_start,append_start_str,append_fin_str))
    return flows


flow_list=specify_flow(edge_dict)