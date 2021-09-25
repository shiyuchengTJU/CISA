#coding=utf-8
from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
from abc import abstractmethod
import logging
import warnings

from foolbox import distances
from foolbox.utils import crossentropy


TAP_or_not = False

if TAP_or_not:
    from new_attack_base import Attack
    from new_attack_base import call_decorator
else:
    from foolbox.attacks.base import Attack
    from foolbox.attacks.base import call_decorator



log_or_not = False
img_save_directory = "/home/xyj/syc/adversarial_machine_learning/nips_2018_competition/nips18/nips18-avc-attack-template__/save_image/"

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def normalize(x):
    return x/np.max(x)


def color_distribute(x, ori_label):
    length = len(x)
    better_color = np.random.rand(length, length)
    color_dict = {ori_label:0}
    new_color_counter = 1
    for i in range(length):
        for j in range(length):
            if x[i][j] == ori_label:  #原始类别，用0表示
                better_color[i][j] = color_dict[ori_label]
            else:
                if x[i][j] in color_dict.keys():
                    better_color[i][j] = color_dict[x[i][j]]
                else:   #没见过的新颜色
                    color_dict[x[i][j]] = new_color_counter
                    better_color[i][j] = new_color_counter
                    new_color_counter+=1 

    return better_color


def cw_distribute(logits, ori_label):
    # print(np.argmax(logits), ori_label)
    if np.argmax(logits)==ori_label:
        temp_logits = logits.copy()
        temp_logits[ori_label] = -999
        return logits[ori_label] - np.max(temp_logits)
    else:
        return  logits[ori_label] - np.max(logits)


def ortho(noise):   #找到一个和当前扰动方向正交的方向，Schmidt
    noise_dim=noise.shape
    # print("dimension of noise is ", noise_dim)

    xr=(np.random.rand(noise_dim[0])*255)
    # print("shape of random vector", np.shape(xr))

    xo=xr-(np.sum(noise*xr)/np.sum(noise**2))*noise
    # print(xo)
    # print(t.max(t.abs(xo)))
    xo/=(np.max(np.abs(xo)))

    xo=np.reshape(xo, (64, 64, 3))

    return xo


def save_picture(img, name):
    # print(img)
    # print(type(img))
    fig=plt.figure(figsize=(40,40))


    output=img
    print(np.shape(output))
    plt.imshow(output)
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(img_save_directory+'classification_region'+name+'.png', bbox_inches='tight')

    # # print(type(img))
    # # scipy.misc.imsave(config["img_save_directory"]+name+'.png', (img.numpy()+1)/2)  #后面这部分算是反归一化
    # im = Image.fromarray(img)
    # im.save(config["img_save_directory"]+name+'.png', (img.numpy()+1)/2)


def save_gradient_picture(mag, pix_num, img, label, name):
    fig=plt.figure(figsize=(60,50))

    # contour map
    plt.axis('off')
    
    subplot = fig.add_subplot(111)

    contour = subplot.contourf(np.arange(pix_num)*mag*2, np.arange(-pix_num/2, pix_num/2), img, cmap="RdBu_r", Nchunk=5)
    # subplot.set_xlim((xs[0], xs[-1]))
    # subplot.set_ylim((ys[0], ys[-1]))
    cb=fig.colorbar(contour)
    cb.ax.tick_params(labelsize=80)


    # #  3d map
    # ax = fig.gca(projection='3d')
    # # as plot_surface needs 2D arrays as input
    # x = np.arange(pix_num)
    # y = np.array(range(pix_num))
    # # we make a meshgrid from the x,y data
    # X, Y = np.meshgrid(x, y)
    # # Z = np.sin(np.sqrt(X**2 + Y**2))

    # # data_value shall be represented by color
    # # map the data to rgba values from a colormap
    # label = normalize(label)
    # colors = cm.ScalarMappable(cmap = "viridis").to_rgba(label)

    # # plot_surface with points X,Y,Z and data_value as colors
    # surf = ax.plot_surface(X, Y, img, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=False)


    plt.savefig(img_save_directory+'gradient'+name+'.png', bbox_inches='tight')   #最后一个参数用来是保存的图片不留空白
    
    # # print(type(img))
    # # scipy.misc.imsave(config["img_save_directory"]+name+'.png', (img.numpy()+1)/2)  #后面这部分算是反归一化
    # im = Image.fromarray(img)
    # im.save(config["img_save_directory"]+name+'.png', (img.numpy()+1)/2)


def draw_gradient(target_model, noise, ori_img, ori_labels, imgName):
    noise = np.random.rand(64, 64, 3)
    noise_dir = np.reshape(noise, (-1))
    ortho_dir = ortho(noise_dir)


    pix_num=50  #画的图横纵轴像素点多少
    mag=8  #倍率

    left=-pix_num
    right=pix_num
    up=-pix_num
    down=pix_num

    temp_row=0
    draw_result=[]
    classification_result = []  #分类结果

    
    
    for row in range(up, down):
        draw_result.append([])
        classification_result.append([])
        for col in range(left, right):
            attack=ori_img+noise*col*mag+ortho_dir*row*mag
            # print("perturb", noise*col*mag+ortho_dir*row*(mag/3))
            attack = np.clip(attack, 0, 255)

            temp_logits, temp_is_adversarial = target_model.predictions(attack)
            loss = crossentropy(ori_labels, temp_logits)

            # draw_result[temp_row].append(loss)
            temp_cw_loss = cw_distribute(softmax(temp_logits), ori_labels)
            draw_result[temp_row].append(temp_cw_loss)  #置信度
            classification_result[temp_row].append(np.argmax(temp_logits))
        # print(draw_result[temp_row])
        print(row)
        temp_row+=1

    np.save("./save_image/", np.array(draw_result))
    # save_gradient_picture(mag, pix_num*2, np.array(draw_result), np.array(color_distribute(classification_result, ori_labels)), str(ori_labels)+'_'+'_'+str(left)+'_'+str(right)+'_'+str(up)+'_'+str(down))
    save_picture(np.array(draw_result), np.array(color_distribute(classification_result, ori_labels)), str(ori_labels)+'_'+'_'+str(left)+'_'+str(right)+'_'+str(up)+'_'+str(down))
    exit()


def draw(target_model, noise, ori_img, ori_labels, adv_labels, imgName, adv_document):
    noise_dir = noise.data.view(-1).cuda()
    ortho_dir = ortho(noise_dir)

    pix_num=50  #画的图横纵轴像素点多少
    mag=8  #倍率

    left=0
    right=pix_num*2
    up=-pix_num
    down=pix_num

    temp_row=0
    draw_result=[]

    class_set = []

    # for row in range(up, down):
    #     draw_result.append([])
    #     for col in range(left, right):
    #         attack=ori_img.data+noise.data*col*0.2+ortho_dir*row*0.2

    #         prediction = target_model(Variable(attack))
    #         _, new_labels = t.max(prediction.data, 1)
    #         # print(prediction_category)

    #         draw_result[temp_row].append(int(new_labels))
    #     # print(draw_result[temp_row])
    #     temp_row+=1

    temp_class_number = 0
    temp_dict={}  #便于提升区分度
    for row in range(up, down):
        draw_result.append([])
        for col in range(left, right):
            attack=ori_img.data+noise.data*col*mag+ortho_dir*row*mag

            if not config['trajectory_or_not']:  #不需要算轨迹
                prediction = target_model(Variable(attack))
                _, new_labels = t.max(prediction.data, 1)
                if new_labels[0] not in class_set:
                    class_set.append(new_labels[0])
                    print(new_labels[0])
                    temp_class_number+=100
                    temp_dict[new_labels[0]]=temp_class_number
                draw_result[temp_row].append(temp_dict[new_labels[0]])

            else:  #需要算轨迹
                draw_result[temp_row].append(0)

            
        # print(draw_result[temp_row])
        print(row)
        temp_row+=1

    if config['trajectory_or_not']:  #需要算轨迹
        temp_number = 100
        for adv_track in adv_document:
            temp_noise = adv_track.data - ori_img.data
            pixel_x = int(t.sum(temp_noise * noise.data)/mag)
            pixel_y = int(t.sum(temp_noise * ortho_dir)/(mag/3))
            print(pixel_x, pixel_y)
            draw_result[pix_num + pixel_y][pixel_x]=temp_number
            temp_number+=100

    save_picture(np.array(draw_result), str(imgName)+'_'+str(ori_labels[0])+'_'+str(adv_labels[0])+'_'+str(left)+'_'+str(right)+'_'+str(up)+'_'+str(down), False)
     





class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, adv_x, class_, strict=True, x=None):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)
        self.success_dir = 0   #对抗样本大致方向之记录
        self.success_adv = 0

        self.best = 9999

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k, scale=scale, bb_step=bb_step, RO=RO, m=m, RC=RC, TAP=TAP, uniform_or_not=uniform_or_not, moment_or_not=moment_or_not)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, scale=scale, bb_step=bb_step, RO=RO, m=m, RC=RC, TAP=TAP, uniform_or_not=uniform_or_not, moment_or_not=moment_or_not)

    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not)

        for i in range(k):
            if try_epsilon(epsilon):
                if log_or_not:
                    logging.info('successful for eps = {}'.format(epsilon))
                break
            if log_or_not:
                logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            if log_or_not:
                logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                if log_or_not:
                    logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                if log_or_not:
                    logging.info('not successful for eps = {}'.format(epsilon))


    def update_success_dir(self, new_adv):
        self.success_adv += new_adv

        new_adv_norm = np.sqrt(np.mean(np.square(self.success_adv)))
        new_adv_norm = max(1e-12, new_adv_norm)
        self.success_dir = self.success_adv/new_adv_norm


    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early, scale, 
                 bb_step=15, RO=False, m=2, RC=False, TAP=False, uniform_or_not=False, moment_or_not=False):
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.original_image.copy()

        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            noise = np.random.uniform(
                -epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype)
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True

        if RC:
            success = False
            momentum_up = 0
            momentum_down = 0
            go_up_flag = True
            x_up = x.copy()

            logits_init, is_adversarial_init = a.predictions(x)
            ce_init = crossentropy(class_, logits_init)
            up_better_start = x.copy()

            for _ in range(iterations):
                avg_gradient_down = 0
                avg_gradient_up = 0
                for m_counter in range(m):
                    #up
                    if RO:
                        if uniform_or_not:
                            temp_x_up = np.clip(np.random.uniform(-scale, scale, original.shape) + x_up + stepsize*self.success_dir, min_, max_).astype(np.float32)
                        else:
                            temp_x_up = np.clip(np.random.normal(loc=x_up, scale=scale) + stepsize*self.success_dir, min_, max_).astype(np.float32)
                    else:
                        if uniform_or_not:
                            temp_x_up = np.clip(np.random.uniform(-scale, scale, original.shape) + x_up, min_, max_).astype(np.float32)
                        else:
                            temp_x_up = np.clip(np.random.normal(loc=x_up, scale=scale), min_, max_).astype(np.float32)
                    temp_x_up.dtype = "float32"
                    if TAP:
                        gradient_up = self._gradient(a, temp_x_up, class_, strict=strict, x=original)
                    else:
                        gradient_up = self._gradient(a, temp_x_up, class_, strict=strict)

                    avg_gradient_up += gradient_up

                    #down
                    if RO:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x + stepsize*self.success_dir, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale) + stepsize*self.success_dir, min_, max_).astype(np.float32)
                    else:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale), min_, max_).astype(np.float32)
                    temp_x_down.dtype = "float32"
                    if TAP:
                        gradient_down = self._gradient(a, temp_x_down, class_, strict=strict, x=original)
                    else:
                        gradient_down = self._gradient(a, temp_x_down, class_, strict=strict)
                    avg_gradient_down += gradient_down
                
                avg_gradient_up = avg_gradient_up/m
                avg_gradient_down = avg_gradient_down/m

                strict = True
                if targeted:
                    gradient_down = -avg_gradient_down
                    gradient_up = -avg_gradient_up

                if moment_or_not:
                    momentum_up += gradient_up
                    momentum_up_norm = np.sqrt(np.mean(np.square(momentum_up)))
                    momentum_up_norm = max(1e-12, momentum_up_norm)  # avoid divsion by zero

                    momentum_down += gradient_down
                    momentum_down_norm = np.sqrt(np.mean(np.square(momentum_down)))
                    momentum_down_norm = max(1e-12, momentum_down_norm)  # avoid divsion by zero
                    if go_up_flag:
                        x_up = x_up - stepsize * (momentum_up/momentum_up_norm)
                    else:
                        x_up = x_up + stepsize * (momentum_up/momentum_up_norm)

                    x = x + stepsize * (momentum_down/momentum_down_norm)

                else: 
                    if go_up_flag:
                        gradient_up = -avg_gradient_up
                        x_up = x_up + stepsize * avg_gradient_up
                    else:
                         x_up = x_up + stepsize * avg_gradient_up

                    x = x + stepsize * avg_gradient_down







                ###  开始画了  ###
                init_dir =  self._clip_perturbation(a, x - original, epsilon)
                draw_gradient(a, init_dir, original, a.original_class, "firstly_first")






                x = original + self._clip_perturbation(a, x - original, epsilon)
                x_up = original + self._clip_perturbation(a, x_up - original, epsilon)

                x = np.clip(x, min_, max_)
                x_up = np.clip(x_up, min_, max_)

                logits_down, is_adversarial_down = a.predictions(x)
                logits_up, is_adversarial_up = a.predictions(x_up)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits_down)
                        logging.debug('crossentropy to {} is {}'.format(
                            a.original_class, ce))
                    ce = crossentropy(class_, logits_down)
                    logging.debug('crossentropy to {} is {}'.format(class_, ce))

                if is_adversarial_up:
                    if RO:
                        self.update_success_dir(x_up)
                    # self.say((np.sum((x_up/255.0 - original/255.0) ** 2))**0.5, "up")
                    #开始二分回溯
                    left = original
                    right = x_up
                    for binary_counter in range(bb_step):
                        middle = np.clip((left + right)/2, min_, max_)
                        temp_logits, temp_is_adversarial = a.predictions(middle)

                        if temp_is_adversarial: #中奖了
                            if RO:
                                self.update_success_dir(middle)
                            # self.say((np.sum((middle/255.0 - original/255.0) ** 2))**0.5, "up")
                            right = middle
                        else:
                            left = middle
                    if return_early:
                        return True
                    else:
                        success = True

                if is_adversarial_down:
                    if RO:
                        self.update_success_dir(x)
                    # self.say((np.sum((x/255.0 - original/255.0) ** 2))**0.5, "down")
                    #开始二分回溯
                    left = original
                    right = x
                    for binary_counter in range(bb_step):
                        middle = np.clip((left + right)/2, min_, max_)
                        temp_logits, temp_is_adversarial = a.predictions(middle)

                        if temp_is_adversarial: #中奖了
                            if RO:
                                self.update_success_dir(middle)
                            # self.say((np.sum((middle/255.0 - original/255.0) ** 2))**0.5, "down")
                            right = middle
                        else:
                            left = middle
                    if return_early:
                        return True
                    else:
                        success = True

    
                if go_up_flag:
                    ce_now = crossentropy(class_, logits_up)
                    if ce_now < ce_init:
                        ce_init = ce_now
                        up_better_start = x_up
                    else:
                        go_up_flag = False
                        momentum_up = 0
                        x_up = up_better_start


        else:    # not roller coaster
            success = False
            momentum_down = 0

            for _ in range(iterations):
                avg_gradient_down = 0
                avg_gradient_up = 0
                for m_counter in range(m):
                    if RO:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x + stepsize*self.success_dir, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale) + stepsize*self.success_dir, min_, max_).astype(np.float32)
                    else:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale), min_, max_).astype(np.float32)
                    temp_x_down.dtype = "float32"
                    if TAP:
                        gradient_down = self._gradient(a, temp_x_down, class_, strict=strict, x=original)
                    else:
                        gradient_down = self._gradient(a, temp_x_down, class_, strict=strict)
                    avg_gradient_down += gradient_down
                
                avg_gradient_down = avg_gradient_down/m

                strict = True
                if targeted:
                    gradient_down = -avg_gradient_down

                if moment_or_not:
                    momentum_down += gradient_down
                    momentum_down_norm = np.sqrt(np.mean(np.square(momentum_down)))
                    momentum_down_norm = max(1e-12, momentum_down_norm)  # avoid divsion by zero
                    x = x + stepsize * (momentum_down/momentum_down_norm)

                else: 
                    x = x + stepsize * avg_gradient_down

                x = original + self._clip_perturbation(a, x - original, epsilon)
                x = np.clip(x, min_, max_)     #保存x轨迹

                document.append(x.clone())

                logits_down, is_adversarial_down = a.predictions(x)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits_down)
                        logging.debug('crossentropy to {} is {}'.format(
                            a.original_class, ce))
                    ce = crossentropy(class_, logits_down)
                    logging.debug('crossentropy to {} is {}'.format(class_, ce))

                if is_adversarial_down:
                    if RO:
                        self.update_success_dir(x)
                    # self.say((np.sum((x/255.0 - original/255.0) ** 2))**0.5, "down")
                    #开始二分回溯
                    left = original
                    right = x
                    for binary_counter in range(bb_step):
                        middle = np.clip((left + right)/2, min_, max_)
                        temp_logits, temp_is_adversarial = a.predictions(middle)

                        if temp_is_adversarial: #中奖了
                            if RO:
                                self.update_success_dir(middle)
                            # self.say((np.sum((middle/255.0 - original/255.0) ** 2))**0.5, "down")
                            right = middle
                        else:
                            left = middle
                    if return_early:
                        return True
                    else:
                        success = True
        return success
        



class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.mean(np.abs(gradient))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, adv_x, class_, strict=True, x=None):
        if x is None:
            gradient = a.gradient(adv_x, class_, strict=strict)
        else:
            gradient = a.gradient(x, adv_x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(np.mean(np.square(gradient)))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning('Running an attack that tries to minimize the'
                            ' Linfinity norm of the perturbation without'
                            ' specifying foolbox.distances.Linfinity as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L1 norm of the perturbation without'
                            ' specifying foolbox.distances.MAE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L2 norm of the perturbation without'
                            ' specifying foolbox.distances.MSE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class LinfinityBasicIterativeAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Basic Iterative Method introduced in [1]_.

    This attack is also known as Projected Gradient
    Descent (PGD) (without random start) or FGMS^k.

    References
    ----------
    .. [1] Alexey Kurakin, Ian Goodfellow, Samy Bengio,
           "Adversarial examples in the physical world",
            https://arxiv.org/abs/1607.02533

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


BasicIterativeMethod = LinfinityBasicIterativeAttack
BIM = BasicIterativeMethod


class L1BasicIterativeAttack(
        L1GradientMixin,
        L1ClippingMixin,
        L1DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L1 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


class L2BasicIterativeAttack(
        L2GradientMixin,
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True,
                 scale = 2,
                 bb_step = 10,
                 RO = False, 
                 m=1, 
                 RC=False, 
                 TAP=False, 
                 uniform_or_not=False, 
                 moment_or_not=False):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not)


class ProjectedGradientDescentAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Projected Gradient Descent Attack
    introduced in [1]_ without random start.

    When used without a random start, this attack
    is also known as Basic Iterative Method (BIM)
    or FGSM^k.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso::

       :class:`LinfinityBasicIterativeAttack` and
       :class:`RandomStartProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=False,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


ProjectedGradientDescent = ProjectedGradientDescentAttack
PGD = ProjectedGradientDescent


class RandomStartProjectedGradientDescentAttack(
        LinfinityGradientMixin,
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Projected Gradient Descent Attack
    introduced in [1]_ with random start.

    References
    ----------
    .. [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
           Dimitris Tsipras, Adrian Vladu, "Towards Deep Learning
           Models Resistant to Adversarial Attacks",
           https://arxiv.org/abs/1706.06083

    .. seealso:: :class:`ProjectedGradientDescentAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=True,
                 return_early=True):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


RandomProjectedGradientDescent = RandomStartProjectedGradientDescentAttack
RandomPGD = RandomProjectedGradientDescent


class MomentumIterativeAttack(
        LinfinityClippingMixin,
        LinfinityDistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """The Momentum Iterative Method attack
    introduced in [1]_. It's like the Basic
    Iterative Method or Projected Gradient
    Descent except that it uses momentum.

    References
    ----------
    .. [1] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su,
           Jun Zhu, Xiaolin Hu, Jianguo Li, "Boosting Adversarial
           Attacks with Momentum",
           https://arxiv.org/abs/1710.06081

    """

    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = \
            self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        return super(MomentumIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 random_start=False,
                 return_early=True):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


MomentumIterativeMethod = MomentumIterativeAttack
