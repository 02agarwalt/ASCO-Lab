import autograd.numpy as np
import numpy.linalg
import numpy.matlib
import math
from scipy.stats import multivariate_normal


class NormalDist:
    """
    Struct to store the params of a normal distribution
    m - mean
    S - covariance
    """

    def __init__(self, m, S):
        assert m.shape[0] == S.shape[0] == S.shape[1]
        self.m = m
        self.S = S
    def sample(self):
        return np.random.multivariate_normal(self.m, self.S)


def renyii_grad(pk0, pk1, a):
    """
    Compute the gradient of the Renyii divergence with respect to the 
    first distribution.  Assumes a diagonal covariance matrix.
    """
    # Check diagonal
    assert(np.all(np.diag(np.diag(pk0.S)) == pk0.S))
    assert(np.all(np.diag(np.diag(pk1.S)) == pk1.S))
    # Check dimensions
    assert(pk0.S.shape == pk1.S.shape)

    Sa = (1 - a) * pk0.S + a * pk1.S
    dm = pk1.m - pk0.m
    drdm = -a * np.dot(dm, np.linalg.inv(Sa))

    diag_Sa = np.diag(Sa)
    diag_p0S = np.diag(pk0.S)

    drdS = -(a / 2.) * dm**2 / (diag_Sa**2) * (1 - a) + \
        ((1-a) / diag_Sa - (1-a) / diag_p0S)/2./(1-a)
    return drdm, drdS


def renyii(pk0, pk1, a):
    """
    Compute the renyii divergence between two Gaussian distributions.
    """
    # Check dimensions
    assert(pk0.S.shape == pk1.S.shape)
    # Check diagonal
    p0S_is_diag = np.all(np.diag(np.diag(pk0.S)) == pk0.S)
    p1S_is_diag = np.all(np.diag(np.diag(pk1.S)) == pk1.S)

    Sa = (1 - a) * pk0.S + a * pk1.S
    # make sure eigenvalues are positive
    if np.any(np.isfinite(Sa) == 0):
	print(Sa)
    w, v = np.linalg.eig(Sa)
    #assert(np.all(w > 0))
    assert np.linalg.det(Sa) != 0
    #if np.linalg.det(Sa) == 0:
    #  print Sa
    #  return float('Inf')

    dm = pk1.m - pk0.m
    # Use precise computation for diagonal covariance matrices
    if p0S_is_diag and p1S_is_diag:
        r = a / 2. * np.dot(np.dot(dm, np.linalg.inv(Sa)), dm) + \
            (np.sum(np.log(np.diag(Sa))) - (1-a)*np.sum(np.log(np.diag(pk0.S))) - a*np.sum(np.log(np.diag(pk1.S)))) \
                / (1 - a) / 2.
    else:
        r = a / 2. * np.dot(np.dot(dm, np.linalg.inv(Sa)), dm) + \
            (np.log(np.linalg.det(Sa)) - (1-a)*np.log(np.linalg.det(pk0.S)) - a*np.log(np.linalg.det(pk1.S))) \
                / (1 - a) / 2.
    #assert(r > -1e-10)
    return max(r, 0)


def is_mvnpdf_baseline(pk, ws, ks, Js):
    """
    Compute the minimum variance unbiased baseline with importance sampling.

    Form given in Tang and Abbeel, On a Connection between Importance
    Sampling and the Likelihood Ratio Policy Gradient, NIPS 2010:

    b = E_Q[\frac{P_\theta(X)}{Q(X)} \grad_\theta P_\theta(X)
            \frac{P_\theta(X)}{Q(X)} \grad_\theta P_\theta(X)^\top]^{-1}
        E_Q[\frac{P_\theta(X)}{Q(X)} \grad_\theta P_\theta(X)
            \frac{P_\theta(X)}{Q(X)} h(X)],
    where h is the function we are baselining, Q is the sampling distribution,
    P is the distribution we are taking expectation over.

    pk - distribution
    pks - sample distributions
    ks - samples
    Js - costs
    """
    d = ks.shape[0]
    L = Js.shape[0]
    M = Js.shape[1]
    assert(ks.shape[2] == L)
    assert(ks.shape[1] == M)

    # Compute gradient of log-likelihood w.r.t. policy parameters
    dm_dlog_ps, dS_dlog_ps = log_mvnpdf_grad(
        np.transpose(ks, axes=[2, 1, 0]), pk.m, np.diag(pk.S))
    grad_log_ps = np.concatenate((dm_dlog_ps, dS_dlog_ps), axis=2)

    # Compute baseline
    outer_grad_log_ps = np.einsum(
        "...i,...j->...ij", grad_log_ps, grad_log_ps.conjugate())
    # make last dimensions L, M for broadcasting, then take sample mean
    b_1 = np.mean(ws**2 * np.transpose(outer_grad_log_ps,
                                       axes=[2, 3, 0, 1]), axis=3)
    b_2 = np.mean(ws**2 * np.transpose(grad_log_ps,
                                       axes=[2, 0, 1]) * Js, axis=2)
    b_1_inv = np.array(map(np.linalg.inv, np.transpose(b_1, axes=[2, 0, 1])))
    b = np.zeros([L, 2 * d])
    # TODO: do this without a loop using numpy if possible
    for i in range(0, L):
        b[i, :] = np.dot(b_1_inv[i, :, :], b_2[:, i])
    #b = np.tensordot(b_1_inv, b_2.transpose(), axes=([1, 2], [1, 1]))
    return b, grad_log_ps

def log_mvnpdf(x, mean, cov):
    k = mean.shape[0]
    offset = x-mean

    return -0.5*(np.log(np.linalg.det(cov)) + k*np.log(2*np.pi)) - \
        0.5*np.sum(np.dot(offset, np.linalg.inv(cov))*offset, axis=-1)

def log_mvnpdf_grad(x, m, S):
    """
    Compute the gradient of the log of a Gaussian pdf w.r.t. its mean and
    covariance given some observation
    x - observation
    m - mean
    S - diagonal covariance elements
    """
    d = m.shape[0]
    assert x.shape[-1] == d
    assert S.shape[0] == d

    ddm = (x - m) / S
    ddS = -1 / (2 * S) + (x - m)**2 / (2 * S**2)
    return ddm, ddS


def mvnpdf_grad(x, m, S):
    """
    Compute the gradient of a Gaussian pdf w.r.t. its mean and
    covariance given some observation
    x - observation
    m - mean
    S - diagonal covariance elements
    """
    d = m.shape[0]
    assert x.shape[-1] == d
    assert S.shape[0] == d

    xl = multivariate_normal.pdf(x, mean=m, cov=np.diag(S))
    if len(x.shape) == 3:
        xl = np.expand_dims(xl, axis=len(xl.shape))
    dpdm = xl * (x - m) / S
    dpdS = xl * (-1 / (2 * S) + (x - m)**2 / (2 * S**2))

    return dpdm, dpdS


def dlogtrunc(x):
    return (1 + x) / (1 + x + 0.5 * x**2)


def logtrunc(x):
    """
    Truncate the higher order moments of x
    """
    return np.log(1 + x + x**2 / 2.)


def dist_bound_robust_cost_func(
    a_and_pk,
    pks,
    Js,
    ks,
    delta,
    Lmax,
    options={
        "analytic_jac": False,
        "normalize_weights": True}):
    L = Js.shape[0]
    assert(len(pks) == L)
    assert(ks.shape[2] == L)
    d = len(pks[0].m)

    a = a_and_pk[0]
    pk = NormalDist(a_and_pk[1:(d + 1)], np.diag(a_and_pk[(d + 1):(2 * d + 1)]))

    if options.get('analytic_jac'):
        Jrob, _, _, Jac = dist_bound_robust(
            a, pk, pks, Js, ks, delta, Lmax, options)
        return (Jrob, np.concatenate(([Jac[2]], Jac[0], Jac[1])))
    else:
        Jrob, _, _ = dist_bound_robust(
            a, pk, pks, Js, ks, delta, Lmax, options)
        return Jrob


def dist_bound_robust(
    a, pk, pks, Js, ks, delta, Lmax, options={
        'analytic_jac': False}):
    """
    Find the bound on J for a policy distribution pk based
    on performances of samples (Js, ks) from past distributions pks

    a - annealing parameter
    delta - bound holds with probability 1-delta
    Lmax - number of past batches to consider from list
    compute_jac - compute the jacobian
    """

    if Lmax < 0:
        i0 = 0
    else:
        i0 = max(0, Js.shape[0] - Lmax)

    pks = pks[i0:]
    Js = Js[i0:, :]
    ks = ks[:, :, i0:]

    M = Js.shape[1]
    L = Js.shape[0]

    assert(len(pks) == L)
    assert(ks.shape[2] == L)

    rdas = np.zeros(L)
    Jmaxs = np.amax(Js, 1)

    # Do everything in log space to avoid numerical issues (like overflow)
    log_ps = log_mvnpdf(np.transpose(
        ks, axes=[2, 1, 0]), mean=pk.m, cov=pk.S)
    log_p0s = np.zeros([L, M])
    for i in range(0, L):
        rdas[i] = renyii(pk, pks[i], 2)
        log_p0s[i, :] = log_mvnpdf(
            ks[:, :, i].transpose(), mean=pks[i].m, cov=pks[i].S)
    if np.any(rdas < 0):
        print("RENYII TERM IS NEGATIVE!!!")

    wss = np.exp(log_ps - log_p0s)
    if options.get('truncate_weights'):
        truncate_thresh = options.get('truncate_thresh')
        if truncate_thresh == None:
            # truncate weights based on 
            # Ionides, Truncated Importance Sampling, 2008
            truncate_thresh = np.sqrt(M)
        wss = np.minimum(wss, truncate_thresh) # truncate weights

    normalization_constant = np.sum(wss, axis=1)
    wss_sum_nonzero_idx = normalization_constant != 0
    # take care of potential divide by zero
    wss[normalization_constant == 0, :] = 1
    if options.get('normalize_weights'):
        # Introduces small bias but reduces variance
        wss[wss_sum_nonzero_idx, :] *= \
            (M / normalization_constant[wss_sum_nonzero_idx])[:, np.newaxis]

    # Weights diagnostics
    M_effective = np.sum(wss, axis=1)**2 / np.sum(wss * wss, axis=1)
    #print("M_effective=\t" + str(M_effective))

    Jrob, Jha, Jh = bound_robust_all(a, wss, Js, rdas, Jmaxs, delta)
    if options.get('analytic_jac'):
        jac = dist_bound_robust_grad(
            a, pk, pks, Js, ks, Jmaxs, rdas, delta, options)
        return Jrob, Jha, Jh, jac
    else:
        return Jrob, Jha, Jh


def bound_robust_all(a, wss, Jss, rdas, Jmaxs, delta):
    M = Jss.shape[1]
    L = Jss.shape[0]

    # robust empirical estimate
    Jha = jha(a, wss, Jss)
    # regularizing term
    divergence_term = np.sum(np.power(Jmaxs, 2) * np.exp(rdas)) * a / 2. / L
    # concentration of measure term
    com_term = math.log(1. / delta) / (a * M * L)
    # compute bound
    Jrob = Jha + divergence_term + com_term
    # compute empirical cost
    Jh = np.sum(Jss * wss) / M

    return Jrob, Jha, Jh


def dist_jha(a, pk, pks, Jss, kss):
    L = Jss.shape[0]
    M = Jss.shape[1]

    wss = np.zeros([L, M])
    for i in range(0, L):
        ps = multivariate_normal.pdf(
            kss[:, :, i].transpose(), mean=pk.m, cov=pk.S)
        p0s = multivariate_normal.pdf(
            kss[:, :, i].transpose(), mean=pks[i].m, cov=pks[i].S)
        ws = ps / p0s
        sum_ws = np.sum(ws)
        if sum_ws == 0:
            ws = np.ones_like(ws)  # avoid divide by zero
        else:
            ws = M * ws / np.sum(ws)
        wss[i, :] = ws
    return jha(a, wss, Jss)

def step_jha(a, ws, rs):
    """
    Step-based robust empirical estimate of expected cost
    """
    M = rs.shape[0]
    step_ws = np.cumprod(ws, axis = 1)
    ls = np.sum(rs*step_ws, axis = 1)
    return np.sum(logtrunc(a * ls))/(M * a)


def jha(a, wss, Jss):
    """
    Compute the robust empirical estimate of expected cost using
    weights wss corresponding to costs Jss
    """

    L = Jss.shape[0]
    M = Jss.shape[1]
    return np.sum(logtrunc(a * Jss * wss)) / (M * L * a)


def dist_jha_grad(a, pk, pks, Jss, kss, options):
    """
    Compute the gradient of the robust expected cost with respect to pk and a
    """
    L = Jss.shape[0]
    M = Jss.shape[1]
    d = pk.m.shape[0]

    assert len(pks) == L
    assert kss.shape[2] == L
    assert kss.shape[1] == M

    kss_trans = np.transpose(kss, axes=[2, 1, 0])
    log_ps = log_mvnpdf(kss_trans, mean=pk.m, cov=pk.S)
    log_p0s = np.zeros((L, M))
    for i in range(0, L):
        log_p0s[i, :] = log_mvnpdf(
            kss[:, :, i].transpose(), mean=pks[i].m, cov=pks[i].S)
    ws = np.exp(log_ps - log_p0s)
    ps = np.exp(log_ps)
    p0s = np.exp(log_p0s)
    if options.get('truncate_weights'):
        truncate_thresh = options.get('truncate_thresh')
        if truncate_thresh == None:
            # truncate weights based on 
            # Ionides, Truncated Importance Sampling, 2008
            truncate_thresh = np.sqrt(M)
        ws_truncated = ws > truncate_thresh
        ws[ws_truncated] = truncate_thresh

    dpdms, dpdSs = mvnpdf_grad(kss_trans, pk.m, np.diag(pk.S))
    if options.get('truncate_weights'):
        dpdms[ws_truncated, :] = 0
        dpdSs[ws_truncated, :] = 0

    if options.get('normalize_weights'):
        sum_ws = np.sum(ws, axis=1)[:, np.newaxis]
        normalized_ws = M * ws / sum_ws

        dlts = dlogtrunc(a * Jss * normalized_ws)

        sum_dwdms = np.sum(dpdms / p0s[:, :, np.newaxis], axis=1)
        sum_dwdSs = np.sum(dpdSs / p0s[:, :, np.newaxis], axis=1)
        dlt_times_J = dlts * Jss
        djdm = np.sum(dlt_times_J[:, :, np.newaxis] * 
            (sum_ws[:, :, np.newaxis] * dpdms / p0s[:, :, np.newaxis] - ws[:, :, np.newaxis] * sum_dwdms[:, np.newaxis, :]) / sum_ws[:, :, np.newaxis]**2, axis=(0, 1)) / L
        djdS = np.sum(dlt_times_J[:, :, np.newaxis] * 
            (sum_ws[:, :, np.newaxis] * dpdSs / p0s[:, :, np.newaxis] - ws[:, :, np.newaxis] * sum_dwdSs[:, np.newaxis, :]) / sum_ws[:, :, np.newaxis]**2, axis=(0, 1)) / L
        Jws = Jss * normalized_ws
    else:
        dlts = dlogtrunc(a * Jss * ws)

        dlt_times_J = dlts * Jss
        djdm = np.sum(dlt_times_J[:, :, np.newaxis] * dpdms /
                      p0s[:, :, np.newaxis], axis=(0, 1)) / (M * L)
        djdS = np.sum(dlt_times_J[:, :, np.newaxis] * dpdSs /
                      p0s[:, :, np.newaxis], axis=(0, 1)) / (M * L)
        Jws = Jss * ws

    djda = -np.sum(logtrunc(a * Jws)) / (a**2 * L * M) + \
        np.sum(dlts * Jws) / (a * L * M)

    return djdm, djdS, djda


def dist_bound_robust_grad(a, pk, pks, Jss, kss, Jmaxs, rdas, delta,
                           options={'normalize_weights': True}):
    L = Jss.shape[0]
    M = Jss.shape[1]
    d = len(pk.m)

    assert len(pks) == L
    assert kss.shape[1] == M
    assert kss.shape[2] == L

    djdm, djdS, djda = dist_jha_grad(a, pk, pks, Jss, kss, options)

    div_terms = (Jmaxs**2 * np.exp(rdas)).ravel()
    drdms = np.zeros((d, L))
    drdSs = np.zeros((d, L))
    for i in range(0, L):
        drdms[:, i], drdSs[:, i] = renyii_grad(pk, pks[i], 2)
    djdS += np.sum(drdSs * a / (2 * L) * div_terms, axis=1)
    djdm += np.sum(drdms * a / (2 * L) * div_terms, axis=1)
    djda += np.sum(div_terms) / (2 * L) - math.log(1. / delta) / (a**2 * L * M)

    return djdm, djdS, djda


def r2_cross(scalar, vector):
    vector_out = np.zeros(2)
    vector_out[0] = -scalar*vector[1]
    vector_out[1] = scalar*vector[0]
    return vector_out

def rot(a):
    return np.array([[math.cos(a), -math.sin(a)],
                  [math.sin(a), math.cos(a)]])
