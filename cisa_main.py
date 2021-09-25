from __future__ import print_function
from struct import unpack
from foolbox.criteria import Misclassification
from straight_model.four_models_straight import create_fmodel_straight
import foolbox
import my_attacks
from boundary.evolutionary_attack import EvolutionaryAttack
from boundary.evolutionary_attack_sample_test import EvolutionaryAttack as EvolutionaryAttack_sample_test
from boundary.bapp import BoundaryAttackPlusPlus as bapp
from boundary.sampling.sample_generator import SampleGenerator
from boundary.perlin import BoundaryAttack as perlin_boundary
from adversarial_vision_challenge import store_adversarial
import sys
import os
from new_composite_model import CompositeModel
import copy
import numpy as np
import torch
import time
import argparse



global marginal_doc 
global doc_len
marginal_doc = np.zeros(301)
doc_len = 0

criterion = foolbox.criteria.Misclassification()


def l2_distance(a, b):
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5


def run_attack_fgsm(model, image, label):
    attack = my_attacks.basic_L2BasicIterativeAttack(model, criterion, )
    return attack(image, label, iterations=1)

def run_attack_fgsm_reverse(model, image, label):
    attack = my_attacks.basic_L2BasicIterativeAttack_reverse(model, criterion, )
    return attack(image, label, iterations=1)

def run_attack_mifgsm(model, image, label, iterations, return_early, binary_search, unpack=True):
    attack = my_attacks.mi_L2BasicIterativeAttack(model, criterion)
    return attack(image, label, iterations=iterations, stepsize=0.05, return_early=return_early, binary_search=binary_search, unpack=unpack)

def run_attack_vr_mifgsm(model, image, label, iterations, return_early, binary_search, unpack=True):
    attack = my_attacks.vr_mi_L2BasicIterativeAttack(model, criterion)
    return attack(image, label, scale=1, iterations=10, binary_search=20, return_early=True, epsilon=0.3, bb_step=0, unpack=unpack)


def run_attack_ifgsm(model, image, label, iterations, return_early, binary_search, unpack=True):
    attack = my_attacks.basic_L2BasicIterativeAttack(model, criterion, )
    return attack(image, label, iterations=iterations, stepsize=0.05, return_early=return_early, binary_search=binary_search, unpack=unpack)


def run_attack_ifgsm_sgd(model, image, label, iterations, stepsize, return_early):
    attack = my_attacks.basic_L2BasicIterativeAttack_sgd(model, criterion, )
    return attack(image, label, iterations=iterations, stepsize=stepsize, return_early=return_early, binary_search=False)


def run_attack_ifgsm_reverse(model, image, label):
    attack = my_attacks.basic_L2BasicIterativeAttack_reverse(model, criterion, )
    return attack(image, label)


def run_attack_ada_ifgsm(model, image, label):
    attack = my_attacks.ada_L2BasicIterativeAttack(model, criterion, )
    return attack(image, label)


def run_attack_omnipotent_fgsm(model, image, label):
    attack = my_attacks.omnipotent_L2BasicIterativeAttack(model, criterion)
    return attack(image, label, scale=1, iterations=5, binary_search=12, return_early=True, 
                  epsilon=0.3, bb_step=3, RO=True, m=1, RC=True, TAP=False, uniform_or_not=False,
                  moment_or_not=False)


def run_attack_gaussian_fgsm(model, image, label, iterations, vr_or_not, scale, m, worthless, binary, RC, exp_step):
    attack = my_attacks.basic_L2BasicIterativeAttack_gaussian(model, criterion, )
    return attack(image, label, iterations=iterations, vr_or_not=vr_or_not, scale=scale, m=m, unpack=False, worthless=worthless, binary=binary, RC=RC, exp_step=exp_step)


def run_attack_cw(model, image, label, cw_binary_search, cw_max_iterations):
    attack = my_attacks.CarliniWagnerL2Attack_rc(model, criterion)
    return attack(image, label, unpack=False, binary_search_steps=cw_binary_search, max_iterations=cw_max_iterations, abort_early=False)
    return attack(image, label, unpack)


def run_attack_ddn(model, image, label, ddn_steps):
    attack = my_attacks.DDNAttack(model, criterion)
    return attack(image, np.array(label), unpack=False, steps=ddn_steps)


def run_attack_deepfool(model, image, label, deepfool_steps):
    attack = foolbox.attacks.DeepFoolL2Attack(model)
    return attack(image, np.array(label), unpack=False, steps=deepfool_steps)

def run_attack_ead(model, image, label, ead_binary_search, ead_max_iterations):
    attack = my_attacks.EADAttack(model)
    return attack(image, np.array(label), unpack=False, binary_search_steps=ead_binary_search, max_iterations=ead_max_iterations, abort_early=False)


def hsja_refinement(model, image, label, hsja_max_query, hsja_starting_point):
    attack = foolbox.attacks.HopSkipJumpAttack(model)
    return attack(image, np.array(label), unpack=False, max_num_evals=1, iterations=int(hsja_max_query/26.0), initial_num_evals=1, starting_point=hsja_starting_point, log_every_n_steps=10000, )

def run_attack_newton(model, image, label, newton_max_iter):
    attack = foolbox.attacks.NewtonFoolAttack(model)

    return attack(image, np.array(label), unpack=False, max_iter=newton_max_iter)

def run_attack_fmna(model, image, label, fmna_max_steps):
    attack = my_attacks.L2FMNAttack(steps = fmna_max_steps)
    higher_image = np.expand_dims(image, axis=0)
    higher_label = np.expand_dims(np.array(label), axis=0)
    return attack.run(model, higher_image, higher_label)

def run_additive(model, image, label, epsilons):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.AdditiveGaussianNoiseAttack(model, criterion)
    return attack(image, label, epsilons=epsilons, unpack=False)


def whey_refinement(image, temp_adv_img, model, label, total_access, first_access, doc_or_not=False, mode='untargeted'):   
    ori_dist = (np.sum((temp_adv_img/255.0 - image/255.0) ** 2))**0.5
    best_dis = ori_dist
    evolutionary_doc = np.zeros(total_access)

    access = 0
    noise = temp_adv_img - image
    for e in range(10):
        for i in range(256, 0, -1):
            noise_temp = copy.deepcopy(noise)
            noise_temp[(noise_temp >= i) & (noise_temp < i+1)] /= 2.0
            noise_temp[(noise_temp > 0) & (noise_temp < 0.5)] = 0

            l2_ori = np.linalg.norm(image/255.0 - (noise+image)/255.0)
            l2_new = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
            if l2_ori - l2_new >= 0.0:
                if (noise != noise_temp).any():
                    access += 1
                    evolutionary_doc[access-1] = best_dis

                    if mode == 'untargeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) != label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            noise = copy.deepcopy(noise_temp)
                    elif mode == 'targeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) == label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            noise = copy.deepcopy(noise_temp)

            noise_temp = copy.deepcopy(noise)
            noise_temp[(noise_temp >= -i-1) & (noise_temp < -i)] /= 2.0
            noise_temp[(noise_temp > -0.5) & (noise_temp < 0)] = 0
            l2_ori = np.linalg.norm(image/255.0 - (noise+image)/255.0)
            l2_new = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
            if l2_ori - l2_new >= 0.0:
                if (noise != noise_temp).any():
                    access += 1
                    evolutionary_doc[access-1] = best_dis
                    if mode == 'untargeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) != label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            noise = copy.deepcopy(noise_temp)
                    elif mode == 'targeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) == label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            noise = copy.deepcopy(noise_temp)

            if access > first_access:
                break
            l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)

    l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)

    while access < total_access:
        evolutionary_doc[access-1] = best_dis
        i, j = int(np.random.random()*60), int(np.random.random()*60)
        noise_temp = copy.deepcopy(noise)
        noise_temp[i:i+3, j:j+3, :] = 0
        l2_ori = np.linalg.norm(image/255.0 - (noise+image)/255.0)
        l2_new = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
        if l2_ori-l2_new >= 0.0:
            access += 1
            if mode == 'untargeted':
                if np.argmax(model.forward_one(np.round(noise_temp + image))) != label:
                    l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                    if l2 < best_dis:
                        best_dis = l2
                    noise = copy.deepcopy(noise_temp)

                elif mode == 'targeted':
                    l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                    if l2 < best_dis:
                        best_dis = l2
                    noise = copy.deepcopy(noise_temp)

        l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)
    
    perturbed_image = noise + image
    l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)


    return perturbed_image, evolutionary_doc


def boundary_refinement(image, temp_adv_img, model, label, total_access, rescale_or_not, source_step=3e-3, spherical_step=1e-1, rate = 0.2, big_size=64, center_size=40, mode='untargeted'):

    initial_time = time.time()
    attacker = EvolutionaryAttack(model)

    temp_result= attacker.attack(image, label, temp_adv_img, initial_time, time_limit=99999999, 
                  iterations=total_access, source_step=source_step, spherical_step=spherical_step, rescale_or_not=rescale_or_not, rate = rate, big_size=big_size, center_size=center_size, mode=mode)
    return temp_result


def boundary_refinement_sample_test(image, temp_adv_img, model, label, total_access, rescale_or_not, source_step=3e-3, spherical_step=1e-1, rate=0.2):

    initial_time = time.time()
    attacker = EvolutionaryAttack_sample_test(model)

    temp_result = attacker.attack(image, label, temp_adv_img, initial_time, time_limit=99999999,
                                  iterations=total_access, source_step=source_step, spherical_step=spherical_step, rescale_or_not=rescale_or_not, rate=rate)
    return temp_result


def perlin_refinement(image, temp_adv_img, model, label, total_access, source_step=3e-3, spherical_step=1e-1, pixels=64):
    attacker = perlin_boundary(model)
    random_generator = SampleGenerator(shape = image.shape, pixels=pixels)

    temp_result = attacker(image, label, starting_point=temp_adv_img, 
                  iterations=total_access, source_step=source_step, spherical_step=spherical_step, sample_gen=random_generator)

    return temp_result


def bapp_refinement(image, temp_adv_img, model, label, initial_num_evals=10, iterations=1, max_num_evals=300):
    criterion = foolbox.criteria.Misclassification()
    attack = bapp(model, criterion)

    temp_result = attack(image, label, starting_point=temp_adv_img,
                         initial_num_evals=initial_num_evals, max_num_evals=max_num_evals, iterations=iterations)

    return temp_result
    

def adversarial_ori_check(adversarial_ori, image, used_iterations, total_access):
    if adversarial_ori is None:   
        return False, 80
    else:   
        temp_dist_ori = l2_distance(adversarial_ori, image)
        if temp_dist_ori > 0:  
            if total_access > used_iterations:  
                return True, total_access - used_iterations
            else:   
                return False, temp_dist_ori

        else:  
            return False, 0


def main(arvg):
    global marginal_doc
    global doc_len


    parser = argparse.ArgumentParser(description='pami')

    parser.add_argument('--dataset', type=str, required=True)  

    parser.add_argument('--TAP_or_not', type=str, default=0)   
    parser.add_argument('--serial_num', type=int, required=True)  
    parser.add_argument('--sub_model_num', type=int, default=1, required=True)
    parser.add_argument('--target_model_num', type=int, default=1, required=True)
    parser.add_argument('--attack_method_num', type=int)  
    parser.add_argument('--total_capacity', type=int, required=True) 
    parser.add_argument('--all_access', type=int, required=True, default=1000)
    parser.add_argument('--whey_or_not', type=int, default=1)   
    parser.add_argument('--total_whey_access', type=int, default=300)  
    parser.add_argument('--first_whey_access', type=int, default=150)  
    parser.add_argument('--boundary_or_not', type=int, default=0)  
    parser.add_argument('--total_boundary_access', type=int, default=1000)  
    parser.add_argument('--boundary_rescale_or_not', type=int, default=0)  
    parser.add_argument('--attention_or_not', type=int, default=0)   
    parser.add_argument('--total_attention_access', type=int, default=300)  
    parser.add_argument('--temp_counter', type=int, default=-1)  
    parser.add_argument('--targeted_mode', type=int, default=0)   
    parser.add_argument('--save_curve_doc', type=int, default=0)   

    parser.add_argument('--IFGSM_stepsize', type=float, default=0.002)   
    parser.add_argument('--IFGSM_return_early', type=int, default=0)   
    parser.add_argument('--IFGSM_iterations', type=int, default=15)   
    parser.add_argument('--IFGSM_binary_search', type=int, default=20)   

    parser.add_argument('--Curls_vr_or_not', type=int, default=1)
    parser.add_argument('--Curls_scale', type=float, default=1.0)
    parser.add_argument('--Curls_m', type=int, default=2)   
    parser.add_argument('--Curls_worthless', type=int, default=1)   
    parser.add_argument('--Curls_binary', type=int, default=0)     
    parser.add_argument('--Curls_RC', type=int, default=1)      

    parser.add_argument('--source_step', type=float, default=3e-3)      
    parser.add_argument('--spherical_step', type=float, default=1e-1)    
    parser.add_argument('--rate', type=float, default=0.2)    
    parser.add_argument('--big_size', type=int, default=64)      
    parser.add_argument('--center_size', type=int, default=40)      
    parser.add_argument('--num_labels', type=int, default=200)     

    parser.add_argument('--init_attack_num', type=int, default=0)      




    args = parser.parse_args()

    if args.dataset == 'TinyImagenet':
        model_dict = {1:"resnet", 2:"inception_small", 3:"inception_resnet", 4:"nasnet", 5:"densenet_adv", 6:"inception_v4_adv", 7:"vgg19_adv", 8:"ensemble_three",}
        from four_models import create_fmodel
        from utils import store_adversarial, compute_MAD, read_images
        from straight_model.four_models_straight import create_fmodel_straight
        
    elif args.dataset == 'Imagenet':
        model_dict = {1:"resnet", 2:"densenet", 3:"vgg", 4:"senet"}
        from four_models_new import create_fmodel
        from utils_imagenet import store_adversarial, compute_MAD, read_images

    elif args.dataset == 'CIFAR':
        model_dict = {1:"vgg16", 2:"resnet"}
        from four_models_cifar import create_fmodel
        from utils_cifar import store_adversarial, compute_MAD, read_images

    elif args.dataset == 'MNIST':
        model_dict = {1:"vgg16", 2:"resnet"}
        from four_models_mnist import create_fmodel
        from utils_mnist import store_adversarial, compute_MAD, read_images


    attack_method_dict = {1:run_attack_fgsm, 
                          2:run_attack_ifgsm, 
                          3:run_attack_mifgsm, 
                          4:run_attack_vr_mifgsm, 
                          5:run_additive,
                          8:run_attack_omnipotent_fgsm,
                          14:run_attack_ada_ifgsm,
                          31:run_attack_fgsm_reverse,
                          32:run_attack_ifgsm_reverse,
                          33:run_attack_gaussian_fgsm,
                          34:run_attack_ifgsm_sgd,
                          35:run_attack_cw,
                          36:run_attack_ddn,
                          37:run_attack_deepfool,
                          38:run_attack_ead,
                          40:run_attack_newton,
                          41:run_attack_fmna,
                          }

    forward_model = create_fmodel(model_dict[args.target_model_num])
    backward_model = create_fmodel(model_dict[args.sub_model_num])

    model = foolbox.models.CompositeModel(
    forward_model=forward_model,
    backward_model=backward_model)

    aux_dist = []
    aux_percent = []
    curve_doc = []
    temp_adv_list = []
    for list_counter in range(args.total_capacity):
        aux_dist.append([]), aux_percent.append([]), curve_doc.append([]), temp_adv_list.append([])

        
    print("serial_num", args.serial_num)
    print("exp_set:", args.sub_model_num, args.target_model_num)


    for (file_name, image, label) in read_images():

        print("---------------------------")
        print(args.temp_counter)


        args.temp_counter += 1


        if args.init_attack_num == 0:
            adversarial_ori_0 = attack_method_dict[2](model, image, label, iterations=1, return_early=False, binary_search=args.all_access)
            total_prediction_calls_0 = args.all_access

            adversarial_ori_1 = attack_method_dict[2](model, image, label, iterations=1, return_early=False, binary_search=args.all_access / 2)
            total_prediction_calls_1 = args.all_access / 2

        elif args.init_attack_num == 1:
            adversarial_ori_unpack_0 = attack_method_dict[2](model, image, label, iterations=args.IFGSM_iterations, return_early=False, binary_search=int(args.all_access/args.IFGSM_iterations), unpack=False)
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls
            
            adversarial_ori_unpack_1 = attack_method_dict[2](model, image, label, iterations=args.IFGSM_iterations, return_early=False, binary_search=args.IFGSM_binary_search, unpack=False)
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 2:
            adversarial_ori_unpack_0 = attack_method_dict[3](model, image, label, iterations=args.IFGSM_iterations, return_early=False, binary_search=int(args.all_access/args.IFGSM_iterations), unpack=False)
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1 = attack_method_dict[3](model, image, label, iterations=args.IFGSM_iterations, return_early=False, binary_search=args.IFGSM_binary_search, unpack=False)
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 3:
            adversarial_ori_unpack_0 = attack_method_dict[4](model, image, label, iterations=args.IFGSM_iterations, return_early=False, binary_search=int(args.all_access/args.IFGSM_iterations), unpack=False)
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1 = attack_method_dict[4](model, image, label, iterations=args.IFGSM_iterations, return_early=False, binary_search=args.IFGSM_binary_search, unpack=False)
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 4:
            adversarial_ori_unpack_1  = attack_method_dict[5](model, image, label, epsilons=int(args.all_access / 10))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 5:
            adversarial_ori_unpack_0  = attack_method_dict[35](model, image, label, 20, int(args.all_access / 20))
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1  = attack_method_dict[35](model, image, label, 20, int(args.all_access / 40))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 6:
            adversarial_ori_unpack_0  = attack_method_dict[36](model, image, label, args.all_access)
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1  = attack_method_dict[36](model, image, label, int(args.all_access / 2))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 7:
            adversarial_ori_unpack_0  = attack_method_dict[37](model, image, label, args.all_access)
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1  = attack_method_dict[37](model, image, label, int(args.all_access / 2))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 8:
            adversarial_ori_unpack_0  = attack_method_dict[38](model, image, label, 20, int(args.all_access / 20))
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1  = attack_method_dict[38](model, image, label, 20, int(args.all_access / 40))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls
        
        elif args.init_attack_num == 9:
            adversarial_ori_unpack_0  = attack_method_dict[40](model, image, label, args.all_access)
            adversarial_ori_0, total_prediction_calls_0 = adversarial_ori_unpack_0._Adversarial__best_adversarial, adversarial_ori_unpack_0._total_prediction_calls

            adversarial_ori_unpack_1  = attack_method_dict[40](model, image, label, int(args.all_access / 2))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        elif args.init_attack_num == 10:
            adversarial_ori_unpack_1  = attack_method_dict[33](model, image, label, iterations=args.all_access, vr_or_not=args.Curls_vr_or_not, scale=args.Curls_scale, m=args.Curls_m, worthless=args.Curls_worthless, binary=args.Curls_binary, RC=args.Curls_RC, exp_step=20)
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls




        check_0, return_0 = adversarial_ori_check(adversarial_ori_1, image, total_prediction_calls_1, args.all_access)
        if check_0:   
            #######################
            temp_adv_list[0] = adversarial_ori_1
            #######################
            aux_dist[0].append(l2_distance(temp_adv_list[0], image))

        else:
            aux_dist[0].append(return_0)

        
        check_1, return_1 = adversarial_ori_check(adversarial_ori_1, image, total_prediction_calls_1, args.all_access)
        if check_1:
            #######################
            temp_adv_list[1], evolutionary_doc_1 = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 2, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[1].append(l2_distance(temp_adv_list[1], image))
        else:
            aux_dist[1].append(return_1)


        if check_1:  
            #######################
            temp_adv_list[2], evolutionary_doc_2 = whey_refinement(image, adversarial_ori_1, model, label, int(return_1), int(return_1//2))
            #######################
            aux_dist[2].append(l2_distance(temp_adv_list[2], image))
        else:
            aux_dist[2].append(return_1)



        if check_1:  
            #######################
            temp_adv_list[3], evolutionary_doc_3 = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 22, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate)
            #######################
            aux_dist[3].append(l2_distance(temp_adv_list[3], image))
        else:
            aux_dist[3].append(return_1)


        if check_1:  
            #######################
            temp_adv_list[4], evolutionary_doc_4 = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 2, source_step=args.source_step, spherical_step=args.spherical_step, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[4].append(l2_distance(temp_adv_list[4], image))
        else:
            aux_dist[4].append(return_1)


        if check_1:   
            #######################
            temp_adv_list[5], evolutionary_doc_5 = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 39, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[5].append(l2_distance(temp_adv_list[5], image))
        else:
            aux_dist[5].append(return_1)

        

        sys.stdout.write("dist of this step:")
        for stdout_counter in range(args.total_capacity):
            print('%.3f' %aux_dist[stdout_counter][-1], end=', ')
        print(' ')

        sys.stdout.write("median dist:")
        for stdout_counter in range(args.total_capacity):
            print('%.3f' %np.median(aux_dist[stdout_counter]), end=', ')
        print(' ')

        sys.stdout.write("mean dist:")
        for stdout_counter in range(args.total_capacity):
            print('%.3f' %np.mean(aux_dist[stdout_counter]), end=', ')
        print(' ')

    print("serial_num", args.serial_num)
    print("exp_set:", sub_model_num, target_model_num)
    np.save('./experiment_result/'+str(args.sub_model_num)+"_"+str(args.target_model_num)+"_"+'.npy', aux_dist)




if __name__ == '__main__':
    main(sys.argv)
