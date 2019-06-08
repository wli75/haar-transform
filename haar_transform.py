import math
import numpy
import os
from scipy.io import wavfile
from scipy import sparse
import matplotlib.pyplot as plt

def to_haar_coeff(u, n):
    for j in range(n-1, -1, -1):
        c = numpy.array(u)
        for i in range(2 ** j):
            sum = numpy.int32(u[2 * i]) + numpy.int32(u[2 * i + 1])
            c[i] = sum / 2
            diff = numpy.int32(u[2 * i]) - numpy.int32(u[2 * i + 1])
            c[2 ** j + i] = diff / 2
        u = c
    return u

def from_haar_coeff(c, n):
    for j in range(n):
        u = numpy.array(c)
        for i in range(2 ** j):
            u[2 * i] = c[i] + c[2 ** j + i]
            u[2 * i + 1] = c[i] - c[2 ** j + i]
        c = u
    return c

def compress_haar_coeff(haar_ceoff, eps):
    haar_coeff_min = numpy.amin(haar_coeff)
    haar_coeff_max = numpy.amax(haar_coeff)
    upper_bound = eps * (numpy.int32(haar_coeff_max) - numpy.int32(haar_coeff_min))
    is_below = numpy.abs(haar_coeff) < upper_bound
    haar_coeff[is_below] = 0
    print(f"set {sum(is_below)} haar coeff < {eps} to 0")

def save_haar_coeff(haar_coef, filename):
    csr_mat = sparse.csr_matrix(haar_coeff)
    sparse.save_npz(filename, csr_mat)

def plot(vec, title):
    plt.figure()
    plt.plot(vec)
    plt.title(title)

from_filename = "handel.wav"
truncated_from_filename = "handel_truncated.wav"
haar_filename = "handel_haar.wav"
haar_vec_filename = "handel_haar.npz"
to_filename = "handel_compressed.wav"
eps = 0.05

rate, vec = wavfile.read(from_filename)
truncated_len = int(math.log2(len(vec)))
truncated_vec = numpy.resize(vec, 2 ** truncated_len)
wavfile.write(truncated_from_filename, rate, truncated_vec)
plot(truncated_vec, "input signal")

haar_coeff = to_haar_coeff(truncated_vec, truncated_len)
plot(haar_coeff, "haar coeff")

compress_haar_coeff(haar_coeff, eps)
plot(haar_coeff, "compressed haar coeff")
save_haar_coeff(haar_coeff, haar_vec_filename)

reconstructed_haar_coeff = sparse.load_npz(haar_vec_filename).toarray()[0]
reconstructed_vec = from_haar_coeff(reconstructed_haar_coeff, truncated_len)
plot(reconstructed_vec, "compressed signal")

wavfile.write(haar_filename, rate, haar_coeff)
wavfile.write(to_filename, rate, reconstructed_vec)

from_filesize = os.stat(truncated_from_filename).st_size
haar_filesize = os.stat(haar_vec_filename).st_size
print(f"original file size = {from_filesize}")
print(f"haar coeff file size = {haar_filesize}")

plt.show()
