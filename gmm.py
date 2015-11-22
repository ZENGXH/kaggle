# gmm.py

import kmeans
import math
#import np
import logging
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
import pdb
import my_io
def pre_kmeans(X,k):
	# X can be unnorlized, which will be done in kmeans
	(normalizeX, r) = kmeans.kmeansClustering(X,k)

	return normalizeX, r


def guassan_distribute(cova, mu, xn, d):
    # d = 2
    # s(np.matrix(xn-mu))
    denomi = math.pow(2*math.pi,2*d) * math.pow(np.linalg.det(cova), 0.5)
    # print denomi
    out = (1.0/ denomi)* math.exp(-0.5*np.matrix(xn-mu)*np.matrix(cova)*np.transpose(np.matrix(xn-mu)))
    return out

def gmm_Esteps(X, pi_k, k, cova_old, mu_old):
	""" E STEP
		retrn pnk = pi_k*N(x|mu_k, con_k)/sum(pi_k*N(x|mu_k, con_k))
	"""
	# compute pi_k = Nk/N
	# compute con_k 
	# compute mu_k


	(N, D) = np.shape(X)
	cova_list = cova_old
	mean_k_list = mu_old
	# for ith group
	# mean_k = np.mean(member_k[i],0)
	# print 'members #',len(member_k[i])

	"""
	diff = [np.subtract(mean_k, member_ki[n]) for n in range(len(member_k[i]))]
	prod = np.multiply(member_ki[:,1],member_ki[:,0])
	conv = np.sum(prod)
	conv = np.divide(conv, N)
	"""

	p = np.zeros((k, N))
	print np.shape(p)
	for n in range(N):
	    # print N
	    pi_N = [pi_k[kk] * guassan_distribute(cova_list[kk], 
	    		mean_k_list[kk], X[n,:], D) for kk in range(k)]
	    deno = np.sum(pi_N)
	    
	    for idx_k in range(k):
	        # print n,idx_k
	        # p[n][idx_k]
	        # p[n][idx_k] = (pi_k[r[n]] * guassan_distribute(cova_list[r[n]], mean_k_list[r[n]], X[n,:], D)) / deno
	        p[idx_k][n] = (pi_k[idx_k] * 
	        				guassan_distribute(cova_list[idx_k], 
	        				mean_k_list[idx_k], X[n,:], D)) / deno
	return p

def gmm_Msteps(mean_k_list, X, p):
	""" M STEP
	"""
	(N, D) = np.shape(X)
	k = len(mean_k_list)

	mu_old = mean_k_list
	mu_new = np.zeros((k,D))
	cova_new = k*[0]
	pi_k_new = np.zeros((k,1))
	N_k = [np.sum(p[idx_k]) for idx_k in range(k)]

	# update mu
	for idx_k in range(k):
	    nomi = [p[idx_k][n]*X[n,:] for n in range(N)]
	    nomi = np.sum(nomi, 0) # DX1
	    deno = N_k[idx_k] # 1X1
	    mu_new[idx_k] = np.divide(nomi,deno)
	    # print np.shape(mu_new[idx_k])
	    
	# update conv
	for idx_k in range(k):
	    deno = N_k[idx_k] 
	    nomi = [ p[idx_k][n]* np.transpose(np.matrix((X[n,:] - mu_old[idx_k]))) * 
	            np.matrix((X[n,:] - mu_old[idx_k])) for n in range(N)] # member in n th group
	    nomi2 = np.sum(nomi,0) # sum vectors in col dimension
	    cova_new[idx_k] = np.divide(np.matrix(nomi2),deno)
	    
	for idx_k in range(k):
	    pi_k_new[idx_k] = (1.0/N) * np.sum(p[idx_k])
	    #if pi_k_new[idx_k] == 0:
	    #	pdb.set_trace()
	return mu_new, cova_new, pi_k_new
"""
def compute_muk(X, K, r):
    my_io.startLog(__name__)
    logger = logging.getLogger(__name__)
    (N, D) = np.shape(X)
    mu = np.zeros(K, D)
    for k in range(0, K):
	    # indexArray = r == k
	    xk = [X[idx,:] for idx in range(len(r)) if r[idx] == k] 
	    # xk: Nk x D
	    logger.info(' in group %d, num of points %s', 
	    			k, str(np.shape(xk)))
	    mu[k,:] = np.mean(xk, 0)  
	    # 0: cal mean along each column
	return mu
"""
def compute_cova(k, X, r):
	#pdb.set_trace()
	member_k = [X[(r==kth)[:,0]] for kth in range(k)]
	# list of length k, each element if the array of the member in that group
	info = [len(member_k[i]) for i in range(k)]
	# logger.info('members:',str(info))
	# print info

	# guans = k*[[0]]
	cova_list = k*[0]
	mean_k_list = k*[0]
	for idx_k in range(k):
		# print idx_k
	    # mean vector of group idx_k , D*1
	    mean_k = np.mean(member_k[idx_k],0) 
	    mean_k_list[idx_k] = mean_k
	    member_ki = member_k[idx_k] # all the member in idx_k th group
	    # print 'members #',len(member_k[i])
	    
	    # now D*N_k

	    cova = np.cov(np.transpose(member_ki))
	    #if idx_k==2:
	    #	pdb.set_trace()
	    # pdb.set_trace() 
	    cova_list[idx_k] = cova
	    # convalence matrix, D*D
	    # logger.info('dimen check cov' if (np.shape(conva)==(D,D)).all else logger.error('dimen check fail'))
	    guans_append = [guassan_distribute(cova, mean_k, member_ki[j], 2) 
	    				for j in range(len(member_ki))]
	    # print type(guans_append)
	    # guans[idx_k] = guans_append # len[4] list, sublist is len[Nk] ie guans[k][n]
	    
	    # np.array(guans_append)
	    # print len(guans)
	return cova_list,mean_k_list

def loss(X, mu, pi_k, cova):
	#Ln = [LA.norm(np.subtract(X[n, :], mu[int(r[n]), :])) 
	#		for k in range(0, N)]
	#Ln_mean = np.mean(Ln)

	k = len(cova)
	(N, D) = np.shape(X)
	Ln = np.zeros((N, 1))
	for n in range(N):
		#pdb.set_trace()
		Ln_n = [pi_k[idx_k]*guassan_distribute(cova[idx_k], 
					mu[idx_k], X[n, :], D) for idx_k in range(k)]
		Ln[n] = np.log(np.sum(Ln_n))
	Ln_sum = np.sum(Ln)

	return math.exp(Ln_sum)*100000000

def gmmClustering(X, k = 2, maxiter = 3):
	my_io.startLog(__name__)
	logger = logging.getLogger(__name__)
	X, r = kmeans.kmeansClustering(X, 2, 1)
	(N, D) = np.shape(X)
	pi_k_old = [np.divide(len(np.where(r==kth)[0]), float(N)) 
			for kth in range(k) ]

	# mu_old = compute_muk(X, k, r)
	cova_old, mu_old = compute_cova(k, X, r)

	for i in range(maxiter):
		#if i==1:
		#	pdb.set_trace()
		logger.info('ite: %d loss: %f',i, 
					loss(X, mu_old, pi_k_old, cova_old)	)	

		p = gmm_Esteps(X, pi_k_old, k, cova_old, mu_old)

		mu_new, cova_new, pi_k_new = gmm_Msteps(mu_old, X, p)
		
		pi_k_old = pi_k_new
		mu_old = mu_new
		cova_old = cova_new
	#matplotlib.cm=get_cmap("jet")
	cm = plt.get_cmap('jet') 
	ax = plt.gca()
    # colors = ['r' if i==0 else 'g' for i in ?]
    #colors = 'r'
    #ax.scatter(X[:,0], X[:,1], c=colors,alpha=0.8)
    # plt.show()
	for j in range(N):
		likehood = p[1][j]
		color = cm(likehood)
		plt.plot(X[j,0], X[j,1]  ,"o", color=color) 
	plt.show()

	for j in range(N):
		likehood = p[0][j]
		color = cm(likehood)
		plt.plot(X[j,0], X[j,1]  ,"o", color=color) 
	plt.show()
	