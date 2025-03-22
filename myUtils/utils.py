import random

import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def cmd_helper(base_str, zip_attr_names:list=None, zip_attr_values:list=None, for_attr_dict:dict=None):
    '''


    :param base_str:
    :param zip_attr_names: [x1, x2, x3]
    :param zip_attr_values: [[v11, v12, v13], [v21, v22, v23]]
    :param for_attr_dict: {x1: {v1, v2}, x2:{v1, v2}}
    :return:
    '''
    if type(base_str) is list:
        results = []
        for bs in base_str:
            results += cmd_helper(bs, zip_attr_names, zip_attr_values, for_attr_dict)
        return results

    target_cmds = []

    zip_cmds = []
    if zip_attr_names:
        assert len(zip_attr_values) > 0, 'zip_attr_values is empty'
        for zavs in zip_attr_values:
            assert len(zavs) == len(zip_attr_names), 'zavs can not be aligned to zip_attr_names'
            zcmd = ''
            for zname, zvalue in zip(zip_attr_names, zavs):
                zcmd += f' {zname} {zvalue}'
            zip_cmds.append(zcmd)

    for_cmds = None
    if for_attr_dict:
        for k, v in for_attr_dict.items():
            if for_cmds is None:
                for_cmds = [f' {k} {vv}' for vv in v]
            else:
                for_cmds_new = []
                for vv in v:
                    for fcmd in for_cmds:
                        for_cmds_new.append(fcmd + f' {k} {vv}')
                for_cmds = for_cmds_new

    if zip_cmds:
        for zcmd in zip_cmds:
            target_cmds.append(base_str + zcmd)

    if for_cmds:
        if len(target_cmds) == 0:
            target_cmds = [base_str]

        new_target_cmds = []
        for fcmd in for_cmds:
            for tcmd in target_cmds:
                new_target_cmds.append(tcmd + fcmd)
        target_cmds = new_target_cmds

    return target_cmds
