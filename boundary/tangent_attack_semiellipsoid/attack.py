import argparse

import os
import random
import sys
sys.path.append(os.getcwd())
sys.path.append("./boundary/tangent_attack_semiellipsoid")
from collections import defaultdict, OrderedDict

import json
from types import SimpleNamespace
import os.path as osp
import torch
import numpy as np
import glog as log
from torch.nn import functional as F
from tangent_import.config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT
from boundary.tangent_attack_semiellipsoid.tangent_point_analytical_solution import TangentFinder as EllipsoidTangentFinder


class EllipsoidTangentAttack(object):
    def __init__(self, model, clip_min, clip_max, height, width, channels, norm, epsilon, radius_ratio,
                 iterations=40, gamma=1.0, stepsize_search='geometric_progression',
                 max_num_evals=1e4, init_num_evals=100, maximum_queries=10000, batch_size=1,
                 verify_tangent_point=False, best_radius=False):
        """
        :param clip_min: lower bound of the image.
        :param clip_max: upper bound of the image.
        :param norm: choose between [l2, linf].
        :param iterations: number of iterations.
        :param gamma: used to set binary search threshold theta. The binary search
                     threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for linf attack.
        :param stepsize_search: choose between 'geometric_progression', 'grid_search'.
        :param max_num_evals: maximum number of evaluations for estimating gradient.
        :param init_num_evals: initial number of evaluations for estimating gradient.
        """
        self.model = model
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.dim = height * width * channels
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (height, width, channels)
        self.radius_ratio = radius_ratio
        self.gamma = 1.0
        if self.norm == "l2":
            self.theta = self.gamma / (np.sqrt(self.dim) * self.dim)
        else:
            self.theta = self.gamma / (self.dim ** 2)
        self.init_num_evals = init_num_evals
        self.max_num_evals = max_num_evals
        self.num_iterations = iterations
        self.gamma = gamma
        self.stepsize_search = stepsize_search
        self.verify_tangent_point = verify_tangent_point

        self.maximum_queries = maximum_queries
        self.batch_size = batch_size
        self.total_images = 1
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.best_radius = best_radius

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        # print("images.shape", images.shape)
        logits = self.model.forward_one(np.round(images[0][0].cpu().numpy()).astype(np.float32))
        logits = torch.from_numpy(logits).unsqueeze(0).cuda()
        if target_labels is None:
            return logits.max(1)[1] != true_labels
        else:
            return logits.max(1)[1] == target_labels

    def decision_function_multi(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        # print("multi images.shape", images.shape)
        logits = self.model.forward(np.round(images.cpu().numpy()).astype(np.float32))

        logits = torch.from_numpy(logits).cuda()

        if target_labels is None:
            return logits.max(1)[1] != true_labels
        else:
            return logits.max(1)[1] == target_labels


    def clip_image(self, image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
        return torch.min(torch.max(image, clip_min), clip_max)


    def project(self, original_image, perturbed_images, alphas):
        alphas_shape = [alphas.size(0)] + [1] * len(self.shape)
        alphas = alphas.reshape(*alphas_shape).cuda()
        if self.norm == 'l2':
            # print("original_image", original_image)
            # print("perturbed_images", perturbed_images)
            return (1 - alphas) * original_image + alphas * perturbed_images
        elif self.norm == 'linf':
            out_images = self.clip_image(perturbed_images, original_image - alphas, original_image + alphas)
            return out_images

    def binary_search_batch(self, original_image, perturbed_images, true_labels, target_labels):
        num_evals = 0
        # Compute distance between each of perturbed image and original image.
        dists_post_update = torch.tensor([
            self.compute_distance(
                original_image,
                perturbed_image,
                self.norm
            ) for perturbed_image in perturbed_images])
        # Choose upper thresholds in binary searchs based on constraint.
        if self.norm == "linf":
            highs = dists_post_update # Stopping criteria.
            thresholds = torch.clamp_max(dists_post_update * self.theta, max=self.theta)
        else:
            highs = torch.ones(perturbed_images.size(0)).cuda()
            thresholds = self.theta
        lows = torch.zeros(perturbed_images.size(0)).cuda()
        # Call recursive function.

        while torch.max((highs - lows) / thresholds).item() > 1:
            # log.info("max in binary search func: {}, highs:{}, lows:{}, highs-lows: {} , threshold {}, (highs - lows) / thresholds: {}".format(torch.max((highs - lows) / thresholds).item(),highs, lows, highs-lows, thresholds, (highs - lows) / thresholds))
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids)
            # Update highs and lows based on model decisions.
            decisions = self.decision_function(mid_images, true_labels, target_labels)
            num_evals += mid_images.size(0)
            decisions = decisions.int()
            # 攻击成功时候high用mid，攻击失败的时候low用mid
            lows = torch.where(decisions == 0, mids, lows)  # lows:攻击失败的用mids，攻击成功的用low
            highs = torch.where(decisions == 1, mids, highs)  # highs: 攻击成功的用mids，攻击失败的用high, 不理解的可以去看论文Algorithm 1
            # log.info("decision: {} low: {}, high: {}".format(decisions.detach().cpu().numpy(),lows.detach().cpu().numpy(), highs.detach().cpu().numpy()))
        out_images = self.project(original_image, perturbed_images, highs)  # high表示classification boundary偏攻击成功一点的线
        # Compute distance of the output image to select the best choice. (only used when stepsize_search is grid_search.)
        dists = torch.tensor([
            self.compute_distance(
                original_image,
                out_image,
                self.norm
            ) for out_image in out_images])
        idx = torch.argmin(dists)
        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist, num_evals

    def select_delta(self, cur_iter, dist_post_update):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.

        """
        if cur_iter == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.norm == 'l2':
                delta = np.sqrt(self.dim) * self.theta * dist_post_update
            elif self.norm == 'linf':
                delta = self.dim * self.theta * dist_post_update
        return delta

    def approximate_gradient(self, sample, true_labels, target_labels, num_evals, delta):
        clip_max, clip_min = self.clip_max, self.clip_min

        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)
        if self.norm == 'l2':
            rv = torch.randn(*noise_shape).cuda()
        elif self.norm == 'linf':
            rv = torch.from_numpy(np.random.uniform(low=-1, high=1, size=noise_shape)).float().cuda()
            # rv = torch.FloatTensor(*noise_shape).uniform_(-1, 1)
        rv = rv / torch.sqrt(torch.sum(torch.mul(rv,rv), dim=(1,2,3),keepdim=True))

        #FIXME 
        temp_shit = delta * rv
        print("sample", sample.shape)
        print("temp_shit", temp_shit.shape)

        perturbed = sample + delta * rv
        perturbed = torch.clamp(perturbed, clip_min, clip_max)
        rv = (perturbed - sample) / delta


        decisions = self.decision_function_multi(perturbed, true_labels, target_labels)
        decision_shape = [decisions.size(0)] + [1] * len(self.shape)

        fval = 2 * decisions.float().reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if torch.mean(fval).item() == 1.0:  # label changes.
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval).item() == -1.0:  # label not change.
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)

        # Get the gradient direction.
        gradf = gradf / torch.norm(gradf,p=2)

        return gradf

    def geometric_progression_for_HSJA(self, x, true_labels, target_labels, update, dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        epsilon = dist.item() / np.sqrt(cur_iter)
        num_evals = np.zeros(1)
        def phi(epsilon, num_evals):
            new = x + epsilon * update
            success = self.decision_function(new[None], true_labels,target_labels)
            num_evals += 1
            return bool(success[0].item())

        while not phi(epsilon, num_evals):  # 只要没有成功，就缩小epsilon
            epsilon /= 2.0
        perturbed = torch.clamp(x + epsilon * update, self.clip_min, self.clip_max)
        return perturbed


    def geometric_progression_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels,
                                                 dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """

        long_radius = dist.item() / np.sqrt(cur_iter)
        short_radius = long_radius / self.radius_ratio
        num_evals = 0

        while True:

            tangent_finder = EllipsoidTangentFinder(x_original.reshape(-1), x_boundary.reshape(-1), short_radius, long_radius, normal_vector.reshape(-1), norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            if self.verify_tangent_point:
                log.info("verifying tagent point")
                another_tangent_point = solve_tangent_point(x_original.reshape(-1).detach().cpu().numpy(), x_boundary.reshape(-1).detach().cpu().numpy(),
                                    normal_vector.reshape(-1).detach().cpu().numpy(), radius, clip_min=self.clip_min,clip_max=self.clip_max)
                if isinstance(another_tangent_point, np.ndarray):
                    another_tangent_point = torch.from_numpy(another_tangent_point).type_as(tangent_point).to(tangent_point.device)
                difference = tangent_point - another_tangent_point
                log.info("Difference max: {:.4f} mean: {:.4f} sum: {:.4f} L2 norm: {:.4f}".format(difference.max().item(),
                                                                                                  difference.mean().item(),
                                                                                                  difference.sum().item(),
                                                                                                  torch.norm(difference)))
            tangent_point = tangent_point.reshape_as(x_original).type(x_original.dtype).unsqueeze(0)
            success = self.decision_function(tangent_point[None], true_labels, target_labels)
            num_evals += 1
            if bool(success[0].item()):
               break
            radius /= 2.0
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point, num_evals


    def log_geometric_progression_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels, dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        radius = dist.item() / np.log2(cur_iter+1)
        num_evals = 0
        while True:

            tangent_finder = TangentFinder(x_original.reshape(-1), x_boundary.reshape(-1), radius, normal_vector.reshape(-1), norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            if self.verify_tangent_point:
                log.info("verifying tagent point")
                another_tangent_point = solve_tangent_point(x_original.reshape(-1).detach().cpu().numpy(), x_boundary.reshape(-1).detach().cpu().numpy(),
                                    normal_vector.reshape(-1).detach().cpu().numpy(), radius, clip_min=self.clip_min,clip_max=self.clip_max)
                if isinstance(another_tangent_point, np.ndarray):
                    another_tangent_point = torch.from_numpy(another_tangent_point).type_as(tangent_point).to(tangent_point.device)
                difference = tangent_point - another_tangent_point
                log.info("Difference max: {:.4f} mean: {:.4f} sum: {:.4f} L2 norm: {:.4f}".format(difference.max().item(),
                                                                                                  difference.mean().item(),
                                                                                                  difference.sum().item(),
                                                                                                  torch.norm(difference)))
            tangent_point = tangent_point.reshape_as(x_original).type(x_original.dtype)
            success = self.decision_function(tangent_point[None], true_labels, target_labels)
            num_evals += 1
            if bool(success[0].item()):
               break
            radius /= 2.0
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point, num_evals


    def fixed_radius_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels, radius):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """

        tangent_finder = TangentFinder(x_original.reshape(-1), x_boundary.reshape(-1), radius, normal_vector.reshape(-1), norm="l2")
        tangent_point = tangent_finder.compute_tangent_point()
        tangent_point = tangent_point.reshape_as(x_original).type(x_original.dtype)
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        success = self.decision_function(tangent_point[None], true_labels, target_labels)
        return tangent_point, bool(success[0].item())

    def binary_search_for_radius_and_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels,
                                                   dist):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        num_evals = 0
        low = 0
        high = dist.item()
        while high - low > 0.1:
            mid = (high + low) / 2.0
            tangent_finder = TangentFinder(x_original.reshape(-1), x_boundary.reshape(-1), mid, normal_vector.reshape(-1),
                                           norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            tangent_point = tangent_point.reshape_as(x_original).type(x_original.dtype)
            success = self.decision_function(tangent_point[None], true_labels, target_labels)[0].item()
            num_evals += 1
            if success:
                high = mid
            else:
                low = mid
        tangent_finder = TangentFinder(x_original.reshape(-1), x_boundary.reshape(-1), high, normal_vector.reshape(-1),
                                       norm="l2")
        tangent_point = tangent_finder.compute_tangent_point()
        tangent_point = tangent_point.reshape_as(x_original).type(x_original.dtype)
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point, num_evals

    def compute_distance(self, x_ori, x_pert, norm='l2'):
        # Compute the distance between two images.
        if norm == 'l2':
            return torch.norm(x_ori - x_pert,p=2).item()
        elif norm == 'linf':
            return torch.max(torch.abs(x_ori - x_pert)).item()

    def attack(self, batch_index, images, target_images, true_labels, target_labels, init_adv):
        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1)*self.batch_size, self.total_images)).tolist()
        assert images.size(0) == 1
        batch_size = images.size(0)
        images = images.squeeze()
        if target_images is not None:
            target_images = target_images.squeeze()
        # Initialize.

        #FIXME
        # perturbed, num_eval = self.initialize(images, target_images, true_labels, target_labels)
        # log.info("after initialize")

        perturbed = init_adv

        dist =  torch.norm((perturbed - images).reshape(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).reshape(-1)

        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        # Project the initialization to the boundary.
        # log.info("before first binary_search_batch")
        perturbed, dist_post_update, num_eval = self.binary_search_batch(images, perturbed[None], true_labels,target_labels)
        # log.info("after first binary_search_batch")

        #FIXME
        # images = images.unsqueeze(0)

        dist =  torch.norm((perturbed - images).reshape(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).reshape(-1)
        query += num_eval
        success_stop_queries[working_ind] = query[working_ind]

        cur_iter = 0
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()
        fixed_radius = dist.item() / np.sqrt(self.num_iterations)
        # init variables
        for j in range(self.num_iterations):
            cur_iter += 1
            # Choose delta.
            delta = self.select_delta(cur_iter, dist_post_update)
            # Choose number of evaluations.
            num_evals = int(self.init_num_evals * np.sqrt(j+1))
            num_evals = int(min([num_evals, self.max_num_evals]))
            # approximate gradient
            gradf = self.approximate_gradient(perturbed, true_labels, target_labels, num_evals, delta)
            if self.norm == "linf":
                gradf = torch.sign(gradf)

            query += num_evals
            # search step size.
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                # if not args.ablation_study: FIXME
                perturbed_Tagent, num_evals = self.geometric_progression_for_tangent_point(images, perturbed, gradf,
                                                                                       true_labels, target_labels,
                                                                                       dist, cur_iter)
                # else:  FIXME
                #     if args.fixed_radius:
                #         perturbed_Tagent, success = self.fixed_radius_for_tangent_point(images, perturbed, gradf, true_labels,
                #                                                                                    target_labels, fixed_radius)
                #         num_evals = torch.zeros_like(query)
                #     elif args.binary_search_radius:
                #         perturbed_Tagent, num_evals = self.binary_search_for_radius_and_tangent_point(images, perturbed, gradf,
                #                                                                                       true_labels, target_labels, dist)
                #     elif args.log2_radius:
                #         perturbed_Tagent, num_evals = self.log_geometric_progression_for_tangent_point(images, perturbed, gradf,
                #                                                                            true_labels, target_labels,
                #                                                                            dist, cur_iter)

                query += num_evals
                # perturbed_HSJA = self.geometric_progression_for_HSJA(perturbed, true_labels, target_labels, gradf, dist, cur_iter)
                # dist_tangent = torch.norm((perturbed_Tagent - images).reshape(batch_size, -1), self.ord, 1).item()
                # dist_HSJA = torch.norm((perturbed_HSJA - images).reshape(batch_size, -1), self.ord, 1).item()
                # log.info("dist of tangent: {:.4f}, dist of HSJA:{:.4f}, tangent < HSJA: {}".format(dist_tangent, dist_HSJA, dist_tangent < dist_HSJA))
                perturbed = perturbed_Tagent
                # Binary search to return to the boundary.
                # log.info("before geometric_progression binary_search_batch")
                perturbed, dist_post_update, num_eval = self.binary_search_batch(images, perturbed[None], true_labels, target_labels)
                # log.info("after geometric_progression binary_search_batch")
                query += num_eval
                dist =  torch.norm((perturbed - images).reshape(batch_size, -1), self.ord, 1)
                working_ind = torch.nonzero(dist > self.epsilon).reshape(-1)
                success_stop_queries[working_ind] = query[working_ind]
                for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                    self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                        inside_batch_index].item()
            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.

                update = gradf
                epsilons = torch.logspace(-4, 0, steps=20) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = torch.clamp(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = self.decision_function(perturbeds, true_labels, target_labels)
                query += perturbeds.size(0)
                for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                    self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                        inside_batch_index].item()
                if idx_perturbed.int().sum().item() > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update, num_eval = self.binary_search_batch(images, perturbeds[idx_perturbed], true_labels, target_labels)
                    query += num_eval
                    dist = torch.norm((perturbed - images).reshape(batch_size, -1), self.ord, 1)
                    working_ind = torch.nonzero(dist > self.epsilon).reshape(-1)
                    success_stop_queries[working_ind] = query[working_ind]
                    for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                        self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                            inside_batch_index].item()
            if torch.sum(query >= self.maximum_queries).item() == true_labels.size(0):
                break
            # if not success:
            #     break
            # compute new distance.
            dist =  torch.norm((perturbed - images).reshape(batch_size, -1), self.ord, 1)
            log.info('{}-th image, iteration: {}, {}: distortion {:.4f}, query: {}'.format(batch_index+1, j + 1, self.norm, dist.item(), int(query[0].item())))
            if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
                break
        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return perturbed, query, success_stop_queries, dist, (dist <= self.epsilon)

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

def EllipsoidTangentAttack_process(image_ori, init_adv, model, label_ori, max_query_num, args):


    temp_height = init_adv.shape[0]
    temp_width = init_adv.shape[1]
    temp_channel = init_adv.shape[2]

    init_adv = torch.from_numpy(init_adv).unsqueeze(0).cuda() 
    image_ori = torch.from_numpy(image_ori).unsqueeze(0).cuda() 
    label_ori = torch.from_numpy(np.array(label_ori)).unsqueeze(0).cuda() 


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args_dict = None

    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000
    if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
        args.max_queries = 20000



    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    # model.cuda()
    # model.eval()
    attacker = EllipsoidTangentAttack(model, 0, 1.0, temp_height, temp_width, temp_channel,
                                    args.norm, args.epsilon, args.radius_ratio, args.num_iterations, gamma=args.gamma, stepsize_search = args.stepsize_search,
                                    max_num_evals=max_query_num,
                                    init_num_evals=args.init_num_eval_grad, maximum_queries=max_query_num,
                                    verify_tangent_point=args.verify_tangent)
    # attacker.attack_all_images(args, arch,  save_result_path)
    result_perturbed, result_query, result_success_stop_queries, result_dist, result_success = attacker.attack(batch_index=0, images=image_ori, true_labels=label_ori, init_adv=init_adv, target_images=None, target_labels=None)

    print("result_perturbed.shape", result_perturbed.shape)

    return result_perturbed[0].cpu().numpy()
