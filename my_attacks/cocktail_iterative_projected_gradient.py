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




def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

class trajectory:
    def __init__(self, x, momentum=None):
        self.img = x
        if momentum is None:
            self.momentum = 0
        else:
            self.momentum = momentum

        self.score = 0
        self.is_adversarial = False



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
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early, scale):

        self.shit = 0

        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 10
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k, scale=scale)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, scale=scale)

    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k, scale):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, scale)

        for i in range(k):
            if try_epsilon(epsilon):
                # logging.info('successful for eps = {}'.format(epsilon))
                break
            # logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            # logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                # logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                # logging.info('not successful for eps = {}'.format(epsilon))

    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early, scale, beam_size=2):
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


        success = False
        momentum = 0
        saved_points = []
        logtis_compare = []
        init_trajectory = trajectory(x)
        saved_points.append(init_trajectory)
        new_points = []  #存储新加入的点

        

        for _ in range(iterations):
            new_points = []
            for x_counter in range(len(saved_points)):

                #vr_MI_FGSM
                temp_x = np.clip(np.random.normal(loc=saved_points[x_counter].img, scale=scale), min_, max_).astype(np.float32)
                temp_x.dtype = "float32"
                ori_new_trajectory = trajectory(saved_points[x_counter].img, saved_points[x_counter].momentum)
                gradient = self._gradient(a, temp_x, class_, strict=strict)
                strict = True
                if targeted:
                    gradient = -gradient

                ori_new_trajectory.momentum += gradient
                momentum_norm = np.sqrt(np.mean(np.square(ori_new_trajectory.momentum)))
                momentum_norm = max(1e-12, momentum_norm)  # avoid divsion by zero
                ori_new_trajectory.img = ori_new_trajectory.img + stepsize * (ori_new_trajectory.momentum/momentum_norm)

                ori_new_trajectory.img = original + self._clip_perturbation(a, ori_new_trajectory.img - original, epsilon)
                ori_new_trajectory.img = np.clip(ori_new_trajectory.img, min_, max_)

                logits, is_adversarial = a.predictions(ori_new_trajectory.img)
                self.shit+=1
                ori_new_trajectory.score = softmax(logits)[class_] + (np.sum((ori_new_trajectory.img/255.0 - original/255.0) ** 2))**0.5

                ori_new_trajectory.is_adversarial = is_adversarial

                # print("vr_MI_FGSM", ori_new_trajectory.score)
                new_points.append(ori_new_trajectory)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        # logging.debug('crossentropy to {} is {}'.format(a.original_class, ce))
                    ce = crossentropy(class_, logits)
                    # logging.debug('crossentropy to {} is {}'.format(class_, ce))
                if is_adversarial:
                    # print(self.shit)
                    if return_early:
                        return True
                    else:
                        success = True


                #MI_FGSM
                ori_new_trajectory = trajectory(saved_points[x_counter].img, saved_points[x_counter].momentum)
                gradient = self._gradient(a, ori_new_trajectory.img, class_, strict=strict)
                strict = True
                if targeted:
                    gradient = -gradient

                ori_new_trajectory.momentum += gradient
                momentum_norm = np.sqrt(np.mean(np.square(ori_new_trajectory.momentum)))
                momentum_norm = max(1e-12, momentum_norm)  # avoid divsion by zero

                ori_new_trajectory.img = ori_new_trajectory.img + stepsize * (ori_new_trajectory.momentum/momentum_norm)

                ori_new_trajectory.img = original + self._clip_perturbation(a, ori_new_trajectory.img - original, epsilon)
                ori_new_trajectory.img = np.clip(ori_new_trajectory.img, min_, max_)

                logits, is_adversarial = a.predictions(ori_new_trajectory.img)
                self.shit+=1
                ori_new_trajectory.score = softmax(logits)[class_] + (np.sum((ori_new_trajectory.img/255.0 - original/255.0) ** 2))**0.5

                ori_new_trajectory.is_adversarial = is_adversarial

                # print("MI-FGSM", ori_new_trajectory.score)
                new_points.append(ori_new_trajectory)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        # logging.debug('crossentropy to {} is {}'.format(a.original_class, ce))
                    ce = crossentropy(class_, logits)
                    # logging.debug('crossentropy to {} is {}'.format(class_, ce))
                if is_adversarial:
                    # print(self.shit)
                    if return_early:
                        return True
                    else:
                        success = True


                #I_FGSM
                ori_new_trajectory = trajectory(saved_points[x_counter].img, saved_points[x_counter].momentum)
                gradient = self._gradient(a, ori_new_trajectory.img, class_, strict=strict)
                strict = True
                if targeted:
                    gradient = -gradient

                gradient_norm = np.sqrt(np.mean(np.square(gradient)))
                gradient_norm = max(1e-12, gradient_norm)  # avoid divsion by zero

                ori_new_trajectory.momentum += gradient
                ori_new_trajectory.img = ori_new_trajectory.img + stepsize * (gradient/gradient_norm)

                ori_new_trajectory.img = original + self._clip_perturbation(a, ori_new_trajectory.img - original, epsilon)
                ori_new_trajectory.img = np.clip(ori_new_trajectory.img, min_, max_)

                logits, is_adversarial = a.predictions(ori_new_trajectory.img)
                self.shit+=1
                ori_new_trajectory.score = softmax(logits)[class_] + (np.sum((ori_new_trajectory.img/255.0 - original/255.0) ** 2))**0.5

                ori_new_trajectory.is_adversarial = is_adversarial

                # print("I-FGSM", ori_new_trajectory.score)
                new_points.append(ori_new_trajectory)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits)
                        # logging.debug('crossentropy to {} is {}'.format(a.original_class, ce))
                    ce = crossentropy(class_, logits)
                    # logging.debug('crossentropy to {} is {}'.format(class_, ce))
                if is_adversarial:
                    # print(self.shit)
                    if return_early:
                        return True
                    else:
                        success = True

                

            ####################  剪枝  #########################
            if _ == 0:    #第一轮
                saved_points = []
            if len(new_points)<=(beam_size+1):
                saved_points.extend(new_points)
                continue
            else:
                saved_points = []
                saved_points.extend(new_points)
                saved_points = sorted(saved_points, key=lambda trajectory_object: trajectory_object.score)
                # print("query time", len(saved_points))
                
                # for temp_sorted_iterator in saved_points:
                #     print(temp_sorted_iterator.score)

                saved_points = saved_points[:beam_size+1]

            # print(self.shit)
         
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
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
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
                 scale = 2):

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
                  random_start, return_early, scale)


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
