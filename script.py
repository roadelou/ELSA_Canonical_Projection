#!/usr/bin/env python3

################################### METADATA ###################################

# Contributors: roadelou
# Contacts:
# Creation Date: 2021-12-25
# Language: Python3
# Python Version: Python 3.10.0

################################### IMPORTS ####################################

# Standard library
from math import cos  # Used for the dot product approximation


# External imports
import numpy as np  # Used for matrixes, for Gramm-Schmidt and the randomness
import matplotlib.pyplot as plt  # Used for the histogram
from matplotlib import colors  # Used for colored plots


# Internal imports
# Your imports within this package go here

################################### CLASSES ####################################

# Your classes go here

################################## FUNCTIONS ###################################


def random_vectors(n):
    """
    Return an array of n random vectors of dimension n.
    """
    return np.random.rand(n, n)


def gramm_schmidt(vectors):
    """
    Turns a matric of random vectors into an orthogonal base of some subset of
    the complete linear space.
    """
    # We perform the QR decomposition with numpy.
    Q, R = np.linalg.qr(vectors)
    # The Q matrix contains the expected orthonormal vectors.
    return Q


def coordinates(vector, base):
    """
    Computes the coordinates of the provided vector in the given base.
    """
    return vector @ base


def sign_vector(vector):
    """
    Reduces the coordinates of a vector to their sign.
    """
    return [1 if vector[0, i] >= 0 else 0 for i in range(np.shape(vector)[1])]


def hamming(a, b):
    """
    Returns the hamming distance of the lists a and b, i.e. their number of
    differences.
    """
    return sum(1 if a[i] != b[i] else 0 for i in range(len(a)))


def norm(vector):
    """
    Returns the norm of the provided vector.
    """
    return np.linalg.norm(vector)


def approx_dot(a, b):
    """
    Dot product approximation used by ELSA accelerator.
    """
    a_sign = sign_vector(a)
    b_sign = sign_vector(b)
    return (
        norm(a)
        * norm(b)
        * cos(np.pi / len(a_sign) * hamming(a_sign, b_sign) - 0.127)
    )


def real_dot(a, b):
    """
    Returns the dot product of the two provided vectors.
    """
    computed = a @ np.transpose(b)
    assert np.shape(computed) == (1, 1)
    return computed[0, 0]


def canonical_evaluation(n, tries=1000):
    """
    Tries to use the canonical base instead of the random one for the hamming
    distance computation.
    """

    ### DATA EXPERIMENT SETUP

    bins = min([max([tries // 10, 20]), 100])
    canonical_base = np.eye(n)
    random_base = gramm_schmidt(random_vectors(n))
    # This list stores the error between two bases for each attempt.
    errors = []
    # We also score the hamming estimate for the canonical base.
    canonical_scores = []
    random_scores = []

    ### DATA GENERATION

    # We generate a bunch of random vectors and see if their hamming distances
    # are the same in the two bases.
    for i in range(tries):
        print(".", end="")
        # We draw two random vectors between -10 and 10.
        a = 20 * np.random.rand(1, n) - 10
        b = 20 * np.random.rand(1, n) - 10
        # We compute the sign vectors.
        a_sign = sign_vector(a)
        b_sign = sign_vector(b)
        # We compute the hamming distance in the canonical base.
        canonical_hamming = hamming(a_sign, b_sign)
        # We project in the random base.
        a_proj = coordinates(a, random_base)
        b_proj = coordinates(b, random_base)
        # We compute the sign vectors.
        a_proj_sign = sign_vector(a_proj)
        b_proj_sign = sign_vector(b_proj)
        # We compute the hamming distance in the random base.
        random_hamming = hamming(a_proj_sign, b_proj_sign)
        # We store the error between the two estimates.
        errors.append(abs(canonical_hamming - random_hamming))
        canonical_scores.append(canonical_hamming)
        random_scores.append(random_hamming)

    ### RESULT ANALYSIS

    errors_sorted = sorted(errors)
    errors_relative = [
        100 * 2 * errors[i] / (canonical_scores[i] + random_scores[i])
        for i in range(tries)
    ]
    errors_relative_sorted = sorted(errors_relative)

    ### ABSOLUTE ERRORS

    # We plot the errors.
    print("\nAbsolute Errors...")
    print("Max: ", max(errors))
    print("Mean: ", np.mean(errors))
    print(
        "Quartiles: ",
        errors_sorted[tries // 4],
        errors_sorted[tries // 2],
        errors_sorted[3 * tries // 4],
    )
    fig_abs, axes_abs = plt.subplots(1, 2)
    axes_abs[0].hist(errors, bins=bins)
    axes_abs[0].set_title(
        "Histogram of Absolute Distance between Canonical and Random Hamming"
    )
    axes_abs[1].plot(range(tries), errors_sorted)
    axes_abs[1].set_title(
        "Progression of Absolute Distance between Canonical and Random Hamming"
    )
    plt.show()

    ### RELATIVE ERRORS

    print("\nRelative Errors...")
    print("Max: ", max(errors_relative))
    print("Mean: ", np.mean(errors_relative))
    print(
        "Quartiles: ",
        errors_relative_sorted[tries // 4],
        errors_relative_sorted[tries // 2],
        errors_relative_sorted[3 * tries // 4],
    )
    fig_rel, axes_rel = plt.subplots(1, 2)
    axes_rel[0].hist(errors_relative, bins=bins)
    axes_rel[0].set_title(
        "Histogram of Relative Distance between Canonical and Random Hamming"
    )
    axes_rel[1].plot(range(tries), errors_relative_sorted)
    axes_rel[1].set_title(
        "Progression of Relative Distance between Canonical and Random Hamming"
    )
    plt.show()


def canonical_evaluation2(n, tries=1000):
    """
    Tries to use the canonical base instead of the random one for the hamming
    distance computation.
    """

    ### DATA EXPERIMENT SETUP

    # Setting variable number of bins.
    bins = min([max([tries // 10, 20]), 100])
    canonical_base = np.eye(n)
    random_base = gramm_schmidt(random_vectors(n))
    # Storing the errors.
    canonical_errors = []
    random_errors = []
    proj_errors = []
    # We also score the estimated dot products.
    canonical_scores = []
    random_scores = []
    ref_scores = []
    proj_scores = []

    ### DATA GENERATION

    # We generate a bunch of random vectors and see if their hamming distances
    # are the same in the two bases.
    for i in range(tries):
        print(".", end="")
        # We draw two random vectors between -10 and 10.
        a = 20 * np.random.rand(1, n) - 10
        b = 20 * np.random.rand(1, n) - 10
        # We compute the reference dot product.
        reference = real_dot(a, b)
        # We compute the canonical approximation.
        canonical_score = approx_dot(a, b)
        # We project in the random base.
        a_proj = coordinates(a, random_base)
        b_proj = coordinates(b, random_base)
        # We compute the dot product of the projection, without approximation.
        proj_score = real_dot(a_proj, b_proj)
        # We compute the random approximation.
        random_score = approx_dot(a_proj, b_proj)
        # We store the error of the two estimates.
        canonical_errors.append(abs(reference - canonical_score))
        random_errors.append(abs(reference - random_score))
        proj_errors.append(abs(reference - proj_score))
        canonical_scores.append(canonical_score)
        random_scores.append(random_score)
        ref_scores.append(reference)
        proj_scores.append(proj_score)

    ### RESULT ANALYSIS

    # Computing the quartiles.
    canonical_sorted = sorted(canonical_errors)
    random_sorted = sorted(random_errors)
    # Computing the relative errors.
    canonical_relative = [
        100 * canonical_errors[i] / abs(ref_scores[i]) for i in range(tries)
    ]
    random_relative = [
        100 * random_errors[i] / abs(ref_scores[i]) for i in range(tries)
    ]
    proj_relative = [
        100 * proj_errors[i] / abs(ref_scores[i]) for i in range(tries)
    ]
    canonical_sorted_relative = sorted(canonical_relative)
    random_sorted_relative = sorted(random_relative)

    ### ABSOLUTE ERRORS

    # We plot the errors.
    print("\nAbsolute Errors...")
    print("Max Canonical: ", max(canonical_errors))
    print("Max Random: ", max(random_errors))
    print("Max Projection: ", max(proj_errors))
    print("Mean Canonical: ", np.mean(canonical_errors))
    print("Mean Random: ", np.mean(random_errors))
    print("Mean Projection: ", np.mean(proj_errors))
    print(
        "Quartiles Canonical:",
        canonical_sorted[tries // 4],
        canonical_sorted[tries // 2],
        canonical_sorted[3 * tries // 4],
    )
    print(
        "Quartiles Random:",
        random_sorted[tries // 4],
        random_sorted[tries // 2],
        random_sorted[3 * tries // 4],
    )
    # Our figures has two rows and two columns of plots.
    fig_abs, axes_abs = plt.subplots(2, 2)
    axes_abs[0][0].hist(canonical_errors, bins=bins)
    axes_abs[0][0].set_title("Canonical Estimation Error Histogram")
    axes_abs[0][1].hist(random_errors, bins=bins)
    axes_abs[0][1].set_title("Random Estimation Error Histogram")
    axes_abs[1][0].hist(proj_errors, bins=bins)
    axes_abs[1][0].set_title("Random Projection Error Histogram")
    # Plotting the sorted scores in order to see the quartiles.
    axes_abs[1][1].plot(
        range(tries), canonical_sorted, "r", label="Canonical Estimation Error"
    )
    axes_abs[1][1].plot(
        range(tries), random_sorted, "b", label="Random Estimation Error"
    )
    axes_abs[1][1].legend()
    axes_abs[1][1].set_title("Error Progression")
    plt.show()

    ### RELATIVE ERRORS

    # Plotting the relative errors.
    print("\nRelative Errors...")
    print("Max Canonical: ", max(canonical_relative))
    print("Max Random: ", max(random_relative))
    print("Max Projection: ", max(proj_relative))
    print("Mean Canonical: ", np.mean(canonical_relative))
    print("Mean Random: ", np.mean(random_relative))
    print("Mean Projection: ", np.mean(proj_relative))
    print(
        "Quartiles Canonical:",
        canonical_sorted_relative[tries // 4],
        canonical_sorted_relative[tries // 2],
        canonical_sorted_relative[3 * tries // 4],
    )
    print(
        "Quartiles Random:",
        random_sorted_relative[tries // 4],
        random_sorted_relative[tries // 2],
        random_sorted_relative[3 * tries // 4],
    )
    # The second figure for the relative errors.
    fig_rel, axes_rel = plt.subplots(2, 2)
    axes_rel[0][0].hist(canonical_relative, bins=bins)
    axes_rel[0][0].set_title("Canonical Estimation Error Histogram")
    axes_rel[0][1].hist(random_relative, bins=bins)
    axes_rel[0][1].set_title("Random Estimation Error Histogram")
    axes_rel[1][0].hist(proj_relative, bins=bins)
    axes_rel[1][0].set_title("Random Projection Error Histogram")
    # Plotting the sorted scores in order to see the quartiles.
    axes_rel[1][1].plot(
        range(tries),
        canonical_sorted_relative,
        "r",
        label="Canonical Estimation Error",
    )
    axes_rel[1][1].plot(
        range(tries),
        random_sorted_relative,
        "b",
        label="Random Estimation Error",
    )
    axes_rel[1][1].set_title("Error Progression")
    axes_rel[1][1].legend()
    plt.show()

    ### PLOTTING SAMPLES

    # Plotting the scores for the first 20 draws.
    plt.plot(range(20), ref_scores[:20], "g", label="Reference dot product")
    plt.plot(
        range(20), canonical_scores[:20], "r", label="Canonical Estimation"
    )
    plt.plot(range(20), random_scores[:20], "b", label="Random Estimation")
    plt.legend()
    plt.title("Comparison of Dot Product Estimators")
    # Same as reference...
    # plt.plot(range(tries), proj_scores, 'cyan')
    plt.show()


##################################### MAIN #####################################

if __name__ == "__main__":
    # We run the two functions to see the statistics of interest.
    canonical_evaluation(64, tries=10**6)
    canonical_evaluation2(64, tries=10**6)

##################################### EOF ######################################
