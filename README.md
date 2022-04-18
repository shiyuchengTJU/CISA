# Query-efficient Black-box Adversarial Attack with Customized Iteration and Sampling (TPAMI'22)
Official implementation of ["**Query-efficient Black-box Adversarial Attack with Customized Iteration and Sampling (TPAMI'22)**"], [Yucheng Shi](https://scholar.google.com/citations?hl=zh-CN&user=annoZWEAAAAJ), [Yahong Han](https://scholar.google.com/citations?hl=zh-CN&user=t4283loAAAAJ), [Qinghua Hu](https://scholar.google.com/citations?hl=zh-CN&user=TVSNq_wAAAAJ), [Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=zh-CN&oi=ao) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN&oi=ao).

> **Abstract:** *It is a challenging task to fool an image classifier based on deep neural networks (DNNs) under the black-box setting where the target model can only be queried. The attacker needs to generate imperceptible adversarial examples with the smallest noise magnitude under a limited number of queries. However, among existing black-box attacks, transfer-based methods tend to overfit the substitute model on parameter settings and iterative trajectories. Decision-based methods have low query efficiency due to fixed sampling and greedy search strategy. To alleviate the above problems, we present a new framework for query-efficient black-box adversarial attack by bridging transfer-based and decision-based attacks. We reveal the relationship between current noise and variance of sampling, the monotonicity of noise compression in decision-based attack, as well as the influence of transition function on the convergence of decision-based attack. Guided by the new framework and theoretical analysis, we propose a black-box adversarial attack named Customized Iteration and Sampling Attack (CISA). CISA estimates the distance from nearby decision boundary to set the stepsize, and uses a dual-direction iterative trajectory to find the intermediate adversarial example. Based on the intermediate adversarial example, CISA conducts customized sampling according to the noise sensitivity of each pixel to further compress noise, and relaxes the state transition function to achieve higher query efficiency. We embed and benchmark existing adversarial attack methods under the new framework. Extensive experiments on several image classification datasets demonstrate CISA's advantage in query efficiency of black-box adversarial attacks.*

<p align="center">
    <img src='/framework.png' width=900/>
</p>

## 1. Environments
Currently, requires following packages
- python 3.7+
- torch 1.7+
- torchvision 0.8+
- CUDA 10.1+
- foolbox 2.0.0rc0



## 2. Usage
### 2.1 Prepare data for attack
The datasets used in the paper are available at the following links:
* [Imagenet](http://image-net.org/index)
* [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/)
* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)

### 2.2 Prepare models
Prepare substitute models and target models. Our model structures mainly follow the [Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch).

### 2.3 Set parameters and attack

Select transfer-based attack for Transfer Attack Module and decision-based attack for Noise Compression Module. Set the attack parameters. For example, run an attack on TinyImagenet with substitute model as resnet, target model as inception_v3, query number as 1000, the transfer-based attack as I-FGSM with 25 binary search and 20 iteration on 10 decision-based attacks, use the following command:

```
python cisa_main.py --serial_num 001 --init_attack_num 2 --sub_model_num 1 --target_model_num 2 --total_capacity 10 --all_access 1000 --dataset TinyImagenet --big_size 64 --center_size 40 --IFGSM_iterations 20 --IFGSM_binary_search 25
```


<!-- ## 3. Citation
TODO -->
