# Robin's attempt at cythonizing this

# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import
import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from mdtraj.utils.six.moves import xrange

__all__ = ['committors', 'conditional_committors',
           '_committors', '_conditional_committors']


def conditional_committors(source, sink, waypoint, msm,
                           forward_committors=None):
    return _conditional_committors(source, sink, waypoint, msm.transmat_,
                                   forward_committors)


@cython.boundscheck(False)
def _conditional_committors(int source, int sink, int waypoint,
                            np.ndarray[DTYPE_t, ndim=2] tprob,
                            double [::1] forward_committors): # TODO nogil
                      #      double [:,::1] tprob,
                      #      np.ndarray[DTYPE_t, ndim=1] forward_committors) nogil:
    """
    Computes the conditional committors :math:`q^{ABC^+}` which are is the
    probability of starting in one state and visiting state B before A while
    also visiting state C at some point.

    Note that in the notation of Dickson et. al. this computes :math:`h_c(A,B)`,
    with ``sources = A``, ``sinks = B``, ``waypoint = C``

    Parameters
    ----------
    waypoint : int
        The index of the intermediate state
    source : int
        The index of the source state
    sink : int
        The index of the sink state
    tprob : np.ndarray
        Transition matrix
    forward_committors : ndarray
        Forward committors source->sink, if pre-calculated

    Returns
    -------
    cond_committors : np.ndarray
        Conditional committors, i.e. the probability of visiting
        a waypoint when on a path between source and sink.

    Notes
    -----
    Employs dense linear algebra, memory use scales as N^2,
    and cycle use scales as N^3

    References
    ----------
    .. [1] Dickson & Brooks (2012), J. Chem. Theory Comput., 8, 3044-3052.
    """

    cdef int n_states = tprob.shape[0]

#    if forward_committors is None:
#        forward_committors = _committors([source], [sink], tprob)

    # permute the transition matrix into cannonical form - send waypoint the the
    # last row, and source + sink to the end after that
    cdef int Bsink_indices[3]
    cdef parma = cvarray(shape=(n_states,), itemsize=sizeof(int),
                         format="i", mode="c")
    cdef int [:] perm = parma

    # "not in Bsink_indices" crashes the cython compiler here...
    # also faster to not use a list comprehension
    Bsink_indices[:] = [source, sink, waypoint]

    cdef int counter = 0
    for i in range(n_states):
        if i != source and i != sink and i != waypoint:
            perm[counter] = i
            counter += 1
    perm[-3:] = Bsink_indices


    # extract P, R
    cdef int n = n_states - 3
    cdef double[:,:] permuted_tprob = tprob[perm, :][:, perm]
    cdef double[:,:] P = permuted_tprob[:n, :n]
    cdef double[:,:] R = permuted_tprob[:n, n:]

    # calculate the conditional committors ( B = N*R ), B[i,j] is the prob
    # state i ends in j, where j runs over the source + sink + waypoint
    # (waypoint is position -1)
    cdef double [:,::1] B = np.dot(np.linalg.inv(np.eye(n) - P), R)

    # add probs for the sinks, waypoint / b[i] is P( i --> {C & not A, B} )
    cdef cond_committors  = np.ndarray(shape=(n_states),
                                       dtype=np.float64)
    #cdef cond_committors = cvarray(shape=(n_states,), itemsize=sizeof(double),
    #                               format="d", mode="c")
    cdef double [:] b = cond_committors
    #for i in range(n):
    #    b[i] = B[i, 2]

    b[:n] = B[:, 2]
    b[n:n+2] = 0.0
    b[n+2] = 1.0

    for i in range(n+3):
        b[i] *= forward_committors[waypoint]

    #cdef double[:] b = B[:, 2] # 3 columns
    #cdef np.ndarray b = np.append(B[:, -1], # TODO BOUNDS
    #                              [0.0] * (len(Bsink_indices) - 1) + [1.0])

    #cdef np.ndarray cond_committors = np.asarray(<np.float32_t[n_states]> b)
    cdef cc = cvarray(shape=(n_states,), itemsize=sizeof(double),
                                   format="d", mode="c")

    # get the original order
    cond_committors = cond_committors[np.argsort(perm)]

    return cond_committors


#def committors(sources, sinks, msm):
#    return _committors(sources, sinks, msm.transmat_)
#
#def _committors(int source, int sink, tprob):
#    """
#    Get the forward committors of the reaction sources -> sinks.
#
#    Parameters
#    ----------
#    sources : int
#        Starting state
#    sinks : int
#        Ending state
#    tprob : np.ndarray
#        Transition matrix
#
#    Returns
#    -------
#    forward_committors : np.ndarray
#        The forward committors for the reaction sources -> sinks
#
#    References
#    ----------
#    .. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
#           transition paths. J. Stat. Phys. 123, 503-523 (2006).
#    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
#           Transition path theory for Markov jump processes.
#           Multiscale Model. Simul. 7, 1192-1219 (2009).
#    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
#           flux and folding pathways in network models of
#           coarse-grained protein dynamics. J. Chem. Phys.
#           130, 205102 (2009).
#    .. [4] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
#           pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
#           19011-19016.
#    """
#    n_states = np.shape(tprob)[0]
#
#    sources = np.array(sources, dtype=int).reshape((-1, 1))
#    sinks = np.array(sinks, dtype=int).reshape((-1, 1))
#
#    # construct the committor problem
#    lhs = np.eye(n_states) - tprob
#
#    for a in sources:
#        lhs[a, :] = 0.0  # np.zeros(n)
#        lhs[:, a] = 0.0
#        lhs[a, a] = 1.0
#
#    for b in sinks:
#        lhs[b, :] = 0.0  # np.zeros(n)
#        lhs[:, b] = 0.0
#        lhs[b, b] = 1.0
#
#    ident_sinks = np.zeros(n_states)
#    ident_sinks[sinks] = 1.0
#
#    rhs = np.dot(tprob, ident_sinks)
#    rhs[sources] = 0.0
#    rhs[sinks] = 1.0
#
#    forward_committors = np.linalg.solve(lhs, rhs)
#
#    return forward_committors
