#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:02:45 2018

@author: Anemometer

Original blog post by Alex Golec:
https://medium.com/@alexgolec/google-interview-questions-deconstructed-the-knights-dialer-impossibly-fast-edition-c288da1685b8
"""

import numpy as np
import copy

# create numpy matrix from upper triangular row list


def create_from_triu(L):
    n = len(L[0])
    A = np.zeros((n, n))
    for i, l in enumerate(L):
        A[i, i:] = np.array(l)
    return A


# the original adjacency matrix, but in
# upper triangular form b/c it is symmetric
NEIGHBORS_TRIU = [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0],
                  [0, 0],
                  [0]]

NEIGHBORS_MATRIX = create_from_triu(NEIGHBORS_TRIU)
NEIGHBORS_MATRIX = NEIGHBORS_MATRIX + NEIGHBORS_MATRIX.T

NEIGHBORS_LIST = NEIGHBORS_MATRIX.tolist()

# Alex's original functions


def matrix_multiply(A, B):
    A_rows, A_cols = len(A), len(A[0])
    B_rows, B_cols = len(B), len(B[0])
    result = list(map(lambda i: [0] * B_cols, range(A_rows)))

    for row in range(A_rows):
        for col in range(B_cols):
            for i in range(B_rows):
                result[row][col] += A[row][i] * B[i][col]

    return result


def count_sequences(start_position, num_hops):
    # Start off with a 10x10 identity matrix
    accum = [[1 if i == j else 0 for i in range(10)] for j in range(10)]

    # bin(num_hops) starts with "0b", slice it off with [2:]
    for bit_num, bit in enumerate(reversed(bin(num_hops)[2:])):
        if bit_num == 0:
            import copy
            power_of_2 = copy.deepcopy(NEIGHBORS_LIST)
        else:
            power_of_2 = matrix_multiply(power_of_2, power_of_2)

        if bit == '1':
            accum = matrix_multiply(accum, power_of_2)

    return matrix_multiply(accum, [[1]]*10)[start_position][0]


# a numpy edition of the same algorithm
def count_sequences_numpy(A, start_position, num_hops):
    n = np.max(np.shape(A))
    # create an nxn identity matrix
    accum = np.eye(n)
    # number of matrix multiplications performed
    num_mults = 0

    # just as in alex' function
    for bit_num, bit in enumerate(reversed(bin(num_hops)[2:])):
        if bit_num == 0:
            import copy
            power_of_2 = copy.deepcopy(NEIGHBORS_MATRIX)
        else:
            power_of_2 = power_of_2.dot(power_of_2)
            num_mults += 1

        if bit == '1':
            accum = accum.dot(power_of_2)
            num_mults += 1

    return num_mults+1, accum.dot(np.ones((n, 1)))[start_position, 0]

# a version that first diagonalizes the input matrix


def count_sequences_diag(A, start_position, num_hops):
    n = np.max(np.shape(A))
    # compute an orthogonal diagonalization of
    # the input matrix; which exists because it
    # is symmetric if it is the adjacecy matrix
    # of an undirected graph
    S, V = np.linalg.eigh(A)

    # start with [1...1], use the fact that
    # A^n = V * (S)^n * V.T
    accum = np.ones((n,))
    S = np.reshape(S, (n,))

    for bit_num, bit in enumerate(reversed(bin(num_hops)[2:])):
        if bit_num == 0:
            power_of_2 = copy.deepcopy(S)
        else:
            power_of_2 = power_of_2 * power_of_2

        if bit == '1':
            accum = accum * power_of_2

    return V.dot(np.diag(accum).dot(V.T)).dot(np.ones(n,))[start_position]


pos = 0
hops = 2
# should be 6
res = np.array([count_sequences(pos, hops),
                count_sequences_numpy(NEIGHBORS_MATRIX, pos, hops)[1],
                count_sequences_diag(NEIGHBORS_MATRIX, pos, hops)])
print('count_sequences, count_sequences_numpy, count_sequences_diag: '
      + str(res))
