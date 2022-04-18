import random
import sys
import os


sys.path.append(os.getcwd())
sys.path.append("./boundary/SignOPT")
import json
from types import SimpleNamespace
import torch
import argparse
import numpy as np
import os.path as osp
import glog as log
from tangent_import.config import MODELS_TEST_STANDARD, IN_CHANNELS
from boundary.SignOPT.sign_opt_l2_norm_attack import SignOptL2Norm


def distance(x_adv, x, norm='l2'):
    diff = (x_adv - x).view(x.size(0), -1)
    if norm == 'l2':
        out = torch.sqrt(torch.sum(diff * diff)).item()
        return out
    elif norm == 'linf':
        out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out

def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.best_initial_target_sample:
        if args.svm:
            if args.attack_defense:
                dirname = 'SVMOPT_best_start_initial_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'SVMOPT_best_start_initial-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            if args.attack_defense:
                dirname = 'SignOPT_best_start_initial_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'SignOPT_best_start_initial-{}-{}-{}'.format(dataset, norm, target_str)
        return dirname
    if args.svm:
        if args.attack_defense:
            dirname = 'SVMOPT_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'SVMOPT-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'SignOPT_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'SignOPT-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def SignOPT_process(model, image_ori, label_ori, max_query_num, args, init_noise, best_distance):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


    image_ori = torch.from_numpy(image_ori).unsqueeze(0).cuda()
    init_noise = torch.from_numpy(init_noise).unsqueeze(0).cuda()  
    label_ori = torch.from_numpy(np.array(label_ori)).unsqueeze(0).cuda() 

    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000

    args.tot = None
    attacker = SignOptL2Norm(model, args.epsilon, args.targeted,
                                args.batch_size, args.est_grad_direction_num,
                            maximum_queries=max_query_num,svm=args.svm,tot=args.tot,
                                best_initial_target_sample=None)
    
    result_perturbed, result_query = attacker.untargeted_attack(images=image_ori, true_labels=label_ori, init_noise=init_noise, best_distance=best_distance)

    # #FIXME
    # print("result_perturbed", result_perturbed.shape)
    # print("result_query", result_query.shape)

    if result_perturbed is not None:
        return result_perturbed[0].cpu().numpy(), result_query.cpu().item() 
    else:
        return None, result_query.cpu().item() 