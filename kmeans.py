import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from numpy import linalg as LA
import my_io
import random
import logging
import pdb
def kmeans_update(X, Mu):
    """update r and Mu given X and Mu
        X is [N D]data
        Mu is [K D] mean vector
        r is 1xN responsibility vector,
        e.g. r = [1,2,1] for 2 clusters 3 data points
        Ln is 1xN minimum distance to its center for each point n
    """

    my_io.startLog(__name__)
    logger = logging.getLogger(__name__)
    
    # initialize
    (K, D1) = np.shape(Mu)
    (N, D) = np.shape(X)
    logger.info('check shape mu: %s X: %s',
                str(np.shape(Mu)),str(np.shape(X)))
    logger.info('pass' if D1==D else logger.error('failed'))
    r = np.zeros((1, N))
    # Mu = zeros(D,K)
    Ln = np.zeros((N,1))
    r = np.zeros((N,1))
    dis2Muk = np.zeros((K,1))

    for n in range(0, N):
        """for each point
          assign x_n to the nearest group
        """
        xn = X[n,:]
        
        # for each cluster, compute the error
        for k in range(0, K):
            """for each group
              calculate the distane from point x_n to mu_k
            """
            dis2Muk[k] = LA.norm(np.subtract(xn, Mu[k,:]))
            # print('dis to cluster %d is %.2f \n',k,dis2Muk(k))
            # np.amin(a, axis=1) return Minima along the second axis
            # (i,j)=np.unravel_index(dis2Muk.argmin(), dis2Muk.shape)

        """ find the minimum distance, ie. the nearest group
            assigh r_n
        """
        indexk = np.argmin(dis2Muk)
        # (minVal, indexK) = min(dis2Muk)
        # assigh cluster
        # compute r

        try:
            r[n] = indexk
        except Exception as err:
            logger.exception('distance to mean vector should be 1D array,'
                             ' now ', np.shape(dis2Muk))
            # fprintf('assign to cluster: %d \n',r(n))
    
    # compute Mu for each k
    """
      for each group:
      update mean vector
    """
    # r is assighment of xk
    # pdb.set_trace()
    for k in range(0, K):
        # indexArray = r == k
        xk = [X[idx,:] for idx in range(len(r)) if r[idx] == k] 
        # xk: Nk x D
        logger.info(' in group %d, num of points %s', k, str(np.shape(xk)))
        Mu[k,:] = np.mean(xk, 0)  # 0: cal mean along each column

    """for each point,
        cal the distance to new mean vector u_k, k = r[n]
    """
    Ln = [LA.norm(np.subtract(X[n, :], Mu[int(r[n]), :])) for n in range(0, N)]
    
    logging.info('cal disortion measurement for each x,'
                 ' check dimension OK' if len(Ln) == n else 'error here')
    # Ln[n] = norm(X[:, n]-Mu[:, r(n)])
    print Ln
    Ln_mean = np.mean(Ln)
    return Ln_mean, r, Mu

"""
def randomInit(maxi, mini, size):
    my_io.startLog(__name__)
    logger = logging.getLogger(__name__)
    random.seed()
    ran = [random.randrange(int(mini), int(maxi)) for i in range(size)]
    logger.info('generate random list'+str(ran)+str(type(ran)))
    return np.array(ran)
"""
def normalize(originx):
    """NOEMALIZE Summary of this function goes here
        normalize data such that they are spread aroung 0
        noticed that the dimension should be N*D 
        ie, normalize each row
        Detailed explanation goes here
    """
    meanX = np.mean(originx, 0)
    stdX = np.std(originx, 0)

    normalizedX = np.array(originx,dtype='f8')

    print meanX 
    print stdX 
    print normalizedX
    for i in range(len(meanX)):
        for j in range(len(originx)):
            normalizedX[j,i] = np.subtract(normalizedX[j,i], meanX[i])
            normalizedX[j,i] = np.divide(normalizedX[j,i], stdX[i])
    return normalizedX

def kmeansClustering(X, k):
    """
        X: data in shape N X D
        k: number of cluster
    """
    my_io.startLog(__name__)
    logger = logging.getLogger(__name__)

    X = normalize(X)
    #X = np.transpose(X)

    (N, D) = np.shape(X)
 
    ax = plt.gca()
    # colors = ['r' if i==0 else 'g' for i in ?]
    colors = 'r'
    ax.scatter(X[:,0], X[:,1], c=colors,alpha=0.8)
    # plt.show()

    # initialize
    # D = np.zeros((0,1))
    min_x = np.amin(X)
    max_x = np.amax(X)
    # shape of mu: K X D
    mu_old = np.zeros((k, D))

    # mu_old = [ randomInit(max_x, min_x, D) for idk in range(k)]
    # mu_old = np.array(mu_old)
    mu_old = np.array(random.sample(X,k))
    logger.info('randomly initialize mean vector, check dimension: ')
    logger.info('pass' if ((k, D) == np.array(mu_old)).all 
                else logger.error('failed'))

    maxIters = 10;
    logger.info('maximum iteration: %s start iteration',
                str(maxIters))

    loss = np.zeros((N,1))
    for i in range(maxIters):
        (Ln, r, Mu) = kmeans_update(X, mu_old)

        loss[i] = Ln
        logger.info('iteration %d, loss %4f \n', i, Ln)

        mu_old = Mu

    logger.info('done loss %3f', Ln)

    ax = plt.gca()
    colors_list = ['r','g','b','y']
    colors = [colors_list[int(r[i])] for i in range(N)]
    # colors = 'r'
    ax.scatter(X[:,0], X[:,1], c=colors,alpha=0.8)
    plt.show()