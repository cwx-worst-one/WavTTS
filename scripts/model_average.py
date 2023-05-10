'''model_average'''
# export PYTHONPATH=.:
import argparse
import os
import os.path as osp
import torch
from core.utils import hdfs_get, hdfs_put, hdfs_mkdir, mkdir_or_exist
from collections import OrderedDict


def parameters_average(checkpoint_list, filename):
    '''parameters_average'''
    state_dict = dict()
    for checkpoint in checkpoint_list:
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state = checkpoint['model']
        else:
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

        for k, v in state.items():
            if state_dict.get(k, None) is not None:
                state_dict[k] += v
            else:
                state_dict[k] = v
    for k in state_dict:
        if state_dict[k].is_floating_point():
            state_dict[k] /= len(checkpoint_list)
        else:
            state_dict[k] //= len(checkpoint_list)
    return state_dict


def load_checkpoint(file_name_list):
    checkpoint_list = []
    for name in file_name_list:
        if not osp.isfile(name):
            raise IOError(f'{name} is not a checkpoint file')
        checkpoint_list.append(torch.load(name, map_location='cpu'))
    state_dict = parameters_average(checkpoint_list, file_name_list)
    save_checkpoint = checkpoint_list[0]

    save_checkpoint['model'] = state_dict
    return save_checkpoint


def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote_save_dir', required=False, default=None)
    parser.add_argument('--local_save_dir', required=False, default='./')
    parser.add_argument('--avg_ckpt_name', required=False, default='average_model.pth')

    parser.add_argument('--resume_ckpts', required=True, help='resume_checkpoints_list')

    args = parser.parse_args()
    return vars(args)


def main():
    '''main'''
    args = parse_args()
    checkpoint_dir = args['local_save_dir']
    path_components = args['resume_ckpts'].split('|')
    if len(path_components) > 1:
        path_root = path_components[0]
        path_list = []
        for fl in path_components[1].split(','):
            path_list.append(osp.join(path_root, fl))
    else:
        path_list = path_components[0].split(',')
    mkdir_or_exist(checkpoint_dir)
    file_name_list = []
    for i, path in enumerate(path_list):
        if path.startswith('hdfs'):
            local_file = osp.join(checkpoint_dir, 'resume_{}.pth'.format(i))
            if hdfs_get(path, local_file) != 0:
                raise RuntimeError("resume file is not exists or local save file is exists")
        else:
            local_file = path
        file_name_list.append(local_file)
    checkpoint = load_checkpoint(file_name_list)
    avg_ckpt = osp.join(checkpoint_dir, args['avg_ckpt_name'])
    with open(avg_ckpt, 'wb') as f:
        torch.save(checkpoint, f)

    print("averager model saved in {}".format(avg_ckpt))
    if args['remote_save_dir']:
        remote_save_dir = args['remote_save_dir']
        hdfs_mkdir(remote_save_dir)
        hdfs_file = osp.join(remote_save_dir, args['avg_ckpt_name'])
        hdfs_put(avg_ckpt, hdfs_file)
        print("averager model saved in {}".format(hdfs_file))



if __name__ == '__main__':
    main()
