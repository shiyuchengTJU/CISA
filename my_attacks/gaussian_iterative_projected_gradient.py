#coding=utf-8

from __future__ import division
import numpy as np
from abc import abstractmethod
import logging
import warnings

from foolbox.attacks.base import Attack
from foolbox.attacks.base import call_decorator
from foolbox import distances
from foolbox.utils import crossentropy


log_or_not = False

def l2_distance(a, b):
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5


def step_value_estimate(noise_g, noise_now, alpha, step_now,):
    #用来判断是否仍有迭代下去的价值
    """Base class for iterative (projected) gradient attacks.
        noise_g    : 高斯噪声幅度
        noise_now  : 迭代到当前这一步已经走过的步数
        alpha      : 使用cab平均每一步实现的噪声压缩，默认0.9985
        step_now   : 当前已经走过的步数
        step_future: 未来还要走的步数
    """
    noise_future = noise_now + noise_now/step_now
    # print("noise_g, noise_now, step_now, current_value", noise_g, noise_now, step_now, noise_future**2 + (1-alpha)*(noise_g**2) - noise_g*noise_future)
    if noise_future**2 + (1-alpha)*(noise_g**2) - noise_g*noise_future >= 0:
        #无意义
        return True
    else:
        return False


def binary_value_estimate(noise_l, noise_r, alpha):
    #用来判断在寻找合适步长时是否仍有二分下去的价值
    """Base class for iterative (projected) gradient attacks.
        noise_l    : 直线上非对抗样本的最大噪声
        noise_r    : 直线上确定是对抗样本的最小噪声（noise_g）
        alpha      : 使用cab平均每一步实现的噪声压缩，默认0.9985
    """

    if  (alpha - 0.75)*noise_r - 0.25*noise_l > 0:
        #值得继续二分
        return True
    else:
        return False




class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, iterations, random_start, return_early, vr_or_not, scale, m, worthless, binary, RC, exp_step):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        a.evolutionary_doc = np.array([])

        min_, max_ = a.bounds()

        targeted, class_ = self._get_mode_and_class(a)

        #首先确定步长
        original = a.unperturbed.copy()

        temp_scale = 1 
        for scale_counter in range(1, 9999):
            temp_scale_x = np.clip(np.random.normal(loc=a.unperturbed, scale=temp_scale), min_, max_).astype(np.float32)
            logits, is_adversarial = a.forward_one(np.round(temp_scale_x))

            if a._Adversarial__best_adversarial is not None:
                a.evolutionary_doc = np.append(a.evolutionary_doc, l2_distance(a._Adversarial__best_adversarial, original))
            else:
                a.evolutionary_doc = np.append(a.evolutionary_doc, 80)

            if is_adversarial:  #成功了，保存噪声幅度
                noise_r = temp_scale_x
                noise_l = original
                
                binary_counter = 0
                if binary:  #选择进行二分
                    for binary_counter in range(1, 9999):
                        dist_r = l2_distance(np.round(noise_r), original)
                        dist_l = l2_distance(np.round(noise_l), original)
                        # print("dist_l, dist_r, estimate", dist_l, dist_r, (0.9985 - 0.75)*dist_r - 0.25*dist_l)
                        if binary_value_estimate(dist_l, dist_r, 0.9985):  #说明值得
                            logits, is_adversarial = a.forward_one(np.round((noise_r + noise_l)/2))
                            if is_adversarial:   #二分成功，出现新的更近的对抗样本
                                noise_r = (noise_r + noise_l)/2
                            else:
                                noise_l = (noise_r + noise_l)/2

                            if a._Adversarial__best_adversarial is not None:
                                a.evolutionary_doc = np.append(a.evolutionary_doc, l2_distance(a._Adversarial__best_adversarial, original))
                            else:
                                a.evolutionary_doc = p.append(a.evolutionary_doc, 80)

                        else:  #不再值得了
                            break
 
                stepsize = l2_distance(noise_r, original)/exp_step
                break
            else:
                temp_scale *= 1.5

            # print("temp_scale", temp_scale)

        #目前已经用了scale_counter+binary_counter次的查询
        return self._run_one(a, stepsize, iterations-scale_counter-binary_counter, random_start, targeted, class_, return_early, vr_or_not, scale, m, l2_distance(temp_scale_x, original), worthless, RC)


    def _run_one(self, a, stepsize, iterations, random_start, targeted, class_, return_early, vr_or_not, scale, m, noise_g, worthless, RC):
        min_, max_ = a.bounds()
        s = max_ - min_

        

        original = a.unperturbed.copy()

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

        success = False

        if RC:  #存在上山下山两条路径，进行初始化
            uphill_worthless_flag = False
            downhill_worthless_flag = False
            uphill_flag = True #上山标志物，True则上，否则下
            uphill_iterations = 0
            downhill_iterations = 0  #上山下山两个方向各走了多少步
            x_uphill = x.copy()
            x_downhill = x.copy()    #上山下山两个维持的对抗样本
            uphill_abandon = False
            downhill_abandon = False    #上山下山两个路径是否放弃
        else:   #只存在一条路径，也就是上山路径
            uphill_worthless_flag = False
            x_uphill = x.copy()

        _ = 0
        for _ in range(iterations):
            if worthless:   #进行价值判断
                if RC:  #在上山/下山场景下进行价值判断
                    if uphill_abandon and downhill_abandon:  #两个方向都放弃了
                        return success, iterations - uphill_iterations - downhill_iterations
                    elif uphill_flag: #目前是上山阶段
                        if uphill_iterations > 0:  #已经不是第一次跑上山路径了
                            uphill_worthless_flag = step_value_estimate(noise_g, l2_distance(x_uphill, original), 0.9985, uphill_iterations)
                        if uphill_iterations > 0 and uphill_worthless_flag == True:    #不值得继续算下去了
                            uphill_abandon = True    #放弃上山
                            uphill_flag = False   #开始下山
                            continue
                    elif uphill_flag == False:   #目前是下山阶段
                        if downhill_iterations > 0:  #已经不是第一次跑下山路径了
                            downhill_worthless_flag = step_value_estimate(noise_g, l2_distance(x_downhill, original), 0.9985, downhill_iterations)
                        if downhill_iterations > 0 and downhill_worthless_flag == True:    #不值得继续算下去了
                            downhill_abandon = True    #放弃下山
                            uphill_flag = True   #开始上山
                            continue
                
                else: #没有上山/下山，单纯的上山路径
                    if _ > 0:  #不是第一步
                        uphill_worthless_flag = step_value_estimate(noise_g, l2_distance(x_uphill, original), 0.9985, _)
                    if _ > 0 and uphill_worthless_flag == True:
                        #不值得继续算下去了
                        return success, iterations - _

            #之后直接用x进行运算
            if RC:
                if uphill_flag:  #目前是上山阶段
                    x = x_uphill 
                else:   #目前是下山阶段
                    x = x_downhill
            else:   #没有上下山的事情
                x = x_uphill


            #使用vr-IGSM来平均梯度
            if vr_or_not:
                avg_gradient = 0
                for m_counter in range(m):
                    temp_x = np.clip(np.random.normal(loc=x, scale=scale), min_, max_).astype(np.float32)
                    temp_x.dtype = "float32"

                    gradient = self._gradient(a, x, class_, strict=strict)
                    avg_gradient += gradient
                
                gradient = avg_gradient/m
            else:
            #不需要vr-IGSM操作
                gradient = self._gradient(a, x, class_, strict=strict)

            # non-strict only for the first call and
            # only if random_start is True
            strict = True
            if targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            
            if RC and uphill_flag == False and downhill_iterations == 0:  
                #走到这里，如果使用上下山，且正在下山，且目前是第一步,则进行一步下山
                gradient = -gradient


            x = x + stepsize * gradient


            x = np.clip(x, min_, max_)

            logits, is_adversarial = a.forward_one(np.round(x))

            if a._Adversarial__best_adversarial is not None:
                a.evolutionary_doc = np.append(a.evolutionary_doc, l2_distance(a._Adversarial__best_adversarial, original))
            else:
                a.evolutionary_doc = np.append(a.evolutionary_doc, 80)

            # #FIXME
            # #查看替代模型与目标模型交叉熵变化
            # backward_logits = a.backward_model_predictions(np.round(x))
            # target_ce = crossentropy(a.original_class, logits)
            # source_ce = crossentropy(a.original_class, backward_logits)
            # print("target_cross_entropy, source_cross_entropy, distance", target_ce, source_ce, l2_distance(np.round(x), original))
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                if targeted:
                    ce = crossentropy(a.original_class, logits)
                    logging.debug('crossentropy to {} is {}'.format(
                        a.original_class, ce))
                ce = crossentropy(class_, logits)
                logging.debug('crossentropy to {} is {}'.format(class_, ce))
            if is_adversarial:
                if return_early:
                    print("final_step_size, iteration", stepsize, _)
                    return True, iterations - _ - 1
                else:
                    success = True

            else:  #如果没有成功，且正在上下山，则更新x_uphill和x_downhill
                if RC:
                    if uphill_flag:    #目前是上山阶段
                        x_uphill = x
                        uphill_iterations += 1
                        
                    elif uphill_flag == False:  #目前是下山阶段
                        x_downhill = x
                        downhill_iterations += 1
                    uphill_flag = (uphill_flag == False)    #转换上/下山
                else:
                    x_uphill = x
                    
        #如果到最后也没成功
        if RC:
            return success, iterations - uphill_iterations - downhill_iterations
        else:    
            return success, iterations - _ - 1


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient_one(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient



class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.mean(np.abs(gradient))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient_one(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf

        # print(np.max(max(1e-12, np.mean(np.square(gradient)))))
        # print(np.min(max(1e-12, np.mean(np.square(gradient)))))
        gradient = gradient / np.sqrt(max(1e-12, np.sum(np.square(gradient))))
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

        self._run(a, iterations, random_start, return_early)


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

        self._run(a, epsilon, stepsize, iterations,
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
    def __call__(self, input_or_adv, label=None, unpack=False,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True,
                 vr_or_not=False,
                 scale=2,
                 m=1,
                 worthless=False,
                 binary=False,
                 RC=False,
                 exp_step=10):
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

        self._run(a, iterations, random_start, return_early, vr_or_not, scale, m, worthless, binary, RC, exp_step)


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

        self._run(a, epsilon, stepsize, iterations,
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

        self._run(a, epsilon, stepsize, iterations,
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
        gradient = a.gradient_one(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = self._decay_factor * self._momentum_history + gradient

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

        self._run(a, epsilon, stepsize, iterations,
                  random_start, return_early)


MomentumIterativeMethod = MomentumIterativeAttack
