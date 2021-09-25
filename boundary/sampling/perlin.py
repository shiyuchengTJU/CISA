"""
Noise generation from Perlin patterns (random and low frequency).

This code is adapted from StackOverflow: https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
And it should be mostly identical to Ken Perlin's reference code @ https://mrl.nyu.edu/~perlin/noise/

We try other noise patterns that we can parametrize, notably simplex noise should be faster.
- see http://webstaff.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
"""

import numpy as np


def calc_fade(n_pixels_side):
    # Precalc fade factors for faster generation

    lin = np.linspace(0, 5, n_pixels_side, endpoint=False)
    x, y = np.meshgrid(lin, lin)

    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    return xi, yi, xf, yf, u, v


def perlin_batch(n_pixels_side, n_samples, precalc_fade=None):
    # Deliver a batch of Perlin noise. This is partially vectorized, so it should be slightly faster.

    if precalc_fade is None:
        precalc_fade = calc_fade(n_pixels_side)
    xi, yi, xf, yf, u, v = precalc_fade

    # Permutation table, which is randomly shuffled on each call.
    p = np.repeat(np.arange(256, dtype=int)[np.newaxis, :], n_samples, axis=0)
    for i in range(n_samples):
        np.random.shuffle(p[i])
    p = np.tile(p, (1, 2))

    # I wonder how to best vectorize this loop.... I really need some students to do this kind of work
    n00 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    n01 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    n11 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    n10 = np.empty((n_samples, n_pixels_side, n_pixels_side), dtype=np.float32)
    for i in range(n_samples):
        # noise components
        n00[i] = gradient(p[i, p[i, xi]+yi],xf,yf)
        n01[i] = gradient(p[i, p[i, xi]+yi+1],xf,yf-1)
        n11[i] = gradient(p[i, p[i, xi+1]+yi+1],xf-1,yf-1)
        n10[i] = gradient(p[i, p[i, xi+1]+yi],xf-1,yf)

    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)        # FIX1: Had to reverse n01 and n10
    return lerp(x1,x2,v)        # FIX2: I also had to reverse x1 and x2 here


def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y


def create_perlin_noise(seed=None, color=True, batch_size=1, normalize=True, precalc_fade=None, pixels = 64):
    """
    Creates a batch of perlin noise patterns.
    :param seed: random seed (optional).
    :param color: if True, generates color noise. If False, generates black-and-white noise (still, the output has 3 identical channels).
    :param batch_size: batch size.
    :param normalize: if True, normalizes each sample to have a L2-norm of 1.
    :param precalc_fade: if not None, uses the provided fade factors for faster calculation.
    :return: samples of shape (b, 64, 64, 3)
    """

    

    if seed is not None:
        np.random.seed(seed)

    output = np.empty((batch_size, pixels, pixels, 3), dtype=np.float32)

    if color:
        batch = perlin_batch(pixels, n_samples=3 * batch_size, precalc_fade=precalc_fade)
        for i in range(batch_size):                     # I'm sure all this could be done in a smarter way
            output[i, :, :, 0] = batch[i * 3]
            output[i, :, :, 1] = batch[i * 3 + 1]
            output[i, :, :, 2] = batch[i * 3 + 2]
    else:
        batch = perlin_batch(pixels, n_samples=batch_size, precalc_fade=precalc_fade)
        for i in range(batch_size):
            output[i, :, :, 0] = batch[i]
            output[i, :, :, 1] = batch[i]
            output[i, :, :, 2] = batch[i]

    # Normalize to L2=1
    if normalize:
        output /= np.linalg.norm(output.reshape(batch_size, -1), axis=1)
    return output