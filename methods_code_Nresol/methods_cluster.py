import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import jv
from scipy.sparse import dok_matrix, csc_matrix
import healpy as hp
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)

#MAIN Functions####################################################
def se_kernel(distrad, length, scale):
    l2 = length ** 2
    return (scale ** 2) * np.exp(-(distrad ** 2) / (2 * l2))


def gaussiankernel_reconstruction_effective_spherical3_blocks(stars, region, length=np.deg2rad(5/60), scale=1.0, no_invvar=False, sigref=None, numblocks=4, return_weights=False, return_numeffstars=False): 
    '''
    stars: Nstar x 5: First 3: unit vector coords, C3: Emedian, C4: Esigma
    region: Npixel x 3: Npixel are the pixels at Nsideresol (=2048, usually)
    length: smoothing length in RADIANS
    scale: height of the kernel at distance 0
    no_invvar: True if you want an error-agnostic smoothing
    sigref: Weight is exp(-rdist^2/(2*l^2))/(sigma_star^2 + sigma_ref^2)
    numblocks: matrix mult subdivisions
    return_weights: return weight matrix
    return_numeffstars: return number of stars that contributed > 1% of the weight matrix for each pixel
    '''
    #BLocks of matrices
    assert stars.shape[1] == 5
    assert region.shape[1] == 3
    train_x = np.hstack([stars[:, 0].reshape((-1, 1)), stars[:, 1].reshape((-1, 1)), stars[:, 2].reshape((-1, 1))])
    train_y = stars[:, 3]
    trainvar = stars[:, 4]**2

    block_indices = np.array_split(np.arange(region.shape[0], dtype=int), numblocks) #ineff since dont need full idxlist
    recon_mean, recon_var = np.zeros(region.shape[0]), np.zeros(region.shape[0])
    
    numeffstars_all = []
    for b in range(numblocks):
        test_x = region[block_indices[b], :]
        cossepmat = np.matmul(test_x, train_x.T) #cos (ang sep): Npix_test x Nstar
        neighbormask = cossepmat > np.cos(5*length)
        wmat = np.zeros(cossepmat.shape)

        if not no_invvar: #no_invvar = False => So inv var weighting
            if sigref is None: #No ref sigma
                for p in range(test_x.shape[0]):
                    wmat[p, neighbormask[p, :]] = se_kernel(np.arccos(np.minimum(cossepmat[p, neighbormask[p, :]], 1.0)), length, scale) / trainvar[neighbormask[p, :]]

            else: #sigref
                sigrefsq = sigref ** 2
                for p in range(test_x.shape[0]):
                    wmat[p, neighbormask[p, :]] = se_kernel(np.arccos(np.minimum(cossepmat[p, neighbormask[p, :]], 1.0)), length, scale) / (trainvar[neighbormask[p, :]] + sigrefsq)
        else: #no Invvar
            for p in range(test_x.shape[0]):
                wmat[p, neighbormask[p, :]] = se_kernel(np.arccos(np.minimum(cossepmat[p, neighbormask[p, :]], 1.0)),
                                                        length, scale)
        #Add tests for pix 0??
        normalization = np.sum(wmat, axis=1).reshape((-1, 1))
        print('No stars for pix', np.sum(np.sum(neighbormask, axis=1)==0)/len(block_indices[b]))
        wmat = wmat / normalization
        #or how many get more than the analytical xsigma weight?
        sig1thresh = se_kernel(length, length, scale) / normalization
        if not no_invvar:
            sig1thresh = sig1thresh / (np.median(trainvar).reshape((-1, 1)) + sigrefsq)
        numeffstars = np.sum(wmat>sig1thresh, axis=1)
        print('Stars Contributing >1sigma/N(pixel) of the weight matrix: Min stars = {}, 16%ile = {:.1f}, median stars = {:.1f}, 84%ile = {:.1f}, max stars = {}'.format(np.min(numeffstars), np.percentile(numeffstars, 16), np.median(numeffstars), np.percentile(numeffstars, 84), np.max(numeffstars)))
        recon_mean[block_indices[b]] = np.matmul(wmat, train_y)
        recon_var[block_indices[b]] = np.matmul(wmat ** 2, trainvar)
        numeffstars_all.append(numeffstars)
    
    if return_weights:
        assert numblocks==1
        if return_numeffstars:
            return recon_mean, recon_var, wmat, numeffstars_all
        else:
            return recon_mean, recon_var, numeffstars_all
    elif return_numeffstars:
        return recon_mean, recon_var, numeffstars_all
    else:
        return recon_mean, recon_var

#GP related + Older code####################################################
class myKernel(torch.nn.Module):
    def __init__(self, lengthscale=None):
        self.training = False
        self.lengthscale=lengthscale

    def forward(self, x1, x2=None):
        raise NotImplementedError

    def __call__(self, x1, x2=None):
        return self.forward(x1, x2)

class ScaledRBFKernel(myKernel):
    def __init__(self, outputscale, lengthscale):
        super().__init__(lengthscale=lengthscale)
        self.outputscale = torch.tensor(outputscale, dtype=torch.float, device=device)

    def forward(self, x1, x2=None): #checked
        assert x1.shape[1]==2
        if x2 is None:
            return torch.tensor(squareform((self.outputscale*torch.exp(-torch.nn.functional.pdist(x1)**2/(2*self.lengthscale**2))).cpu().numpy()) + self.outputscale.cpu().numpy()*np.eye(x1.shape[0]), device=device, dtype=torch.float) #you have to add the torch.eye because the diagonal is 0 otherwise
        else:
            assert x2.shape[1] == 2
            return torch.tensor(self.outputscale*torch.exp(- torch.cdist(x1, x2, p=2)**2 / (2*self.lengthscale**2)), dtype=torch.float)


def SqExpBasis(x, length, scale):
    return (scale ** 2) * np.exp(-(x ** 2 / (2 * (length ** 2))))


def torchSqExpBasis(x, length, scale):
    return (scale ** 2) * torch.exp(-torch.square(x) / (2 * (length ** 2)))


def se2d_equivalentkernel(rho, sigma, length):
    b = sigma ** 2 / (rho * (2 * np.pi * (length ** 2)))
    sc = np.sqrt(np.log(1.0 / b) / (2 * (np.pi ** 2) * (length ** 2)))

    def hse(r):
        return (sc / r) * jv(1, 2 * np.pi * sc * r)

    return hse



def gp_reconstruction(stars, region, length=5.0, scale=1.0, diagreg='Sigma', verbose=False, return_posterr=True):
    train_x = np.hstack([stars[:, 0].reshape((-1, 1)), stars[:, 1].reshape((-1, 1))])
    train_y = stars[:, 2]
    test_x = region
    train_ysig = stars[:, 3]

    pwise = squareform(pdist(train_x))  # slowest
    cov_train = SqExpBasis(pwise, length, scale)
    print('Pre reg CN = ', np.linalg.cond(cov_train))
    cov_test_train = SqExpBasis(cdist(test_x, train_x), length, scale)
    if diagreg == 'Sigma':
        diagreg = np.diag(train_ysig ** 2)
    else:
        diagreg = diagreg
    print('Post Reg CondNo=', np.linalg.cond(cov_train + diagreg))
    cov_trinv = np.linalg.inv(cov_train + diagreg)

    gprecon = np.mean(train_y) + (
        np.matmul(np.matmul(cov_test_train, cov_trinv), (train_y.reshape((len(train_x), 1)) - np.mean(train_y))))
    if verbose:
        return gprecon, cov_train
    if return_posterr:
        gp_postcov = SqExpBasis(squareform(pdist(test_x)), length, scale) - np.matmul(
            np.matmul(cov_test_train, cov_trinv),
            SqExpBasis(cdist(train_x, test_x), length, scale))  # probably gonna crash
    else:
        return gprecon
    return gprecon, gp_postcov


def gp_reconstruction_pytorch(stars, region, length=np.deg2rad(1), scale=1.0, diagreg='Sigma', verbose=False, return_posterr=True,
                              train_frac=1.0, kernel='RBF', nu=None, gp_mean=None, check_CN=True):
    #this assumed you'd already projected into 2D
    #All arguments should then be in radians
    if kernel=='RBF':
        kernfunc = ScaledRBFKernel(outputscale=scale, lengthscale=length)
    else:
        assert nu is not None
        raise NotImplementedError
        #matern
    assert stars.shape[1]==4
    assert region.shape[1]==2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if train_frac == 1.0:
        ttrain_x = torch.tensor(np.hstack([stars[:, 0].reshape((-1, 1)), stars[:, 1].reshape((-1, 1))]), device=device)
        ttrain_y = torch.tensor(stars[:, 2], device=device)
        train_sig2 = stars[:, 3] ** 2

    else:
        choice_idx = np.random.choice(stars.shape[0], size=int(train_frac * stars.shape[0]), replace=False)
        ttrain_x = torch.tensor(
            np.hstack([stars[choice_idx, 0].reshape((-1, 1)), stars[choice_idx, 1].reshape((-1, 1))]), device=device)
        ttrain_y = torch.tensor(stars[choice_idx, 2], device=device)
        train_sig2 = stars[choice_idx, 3] ** 2

    ttest_x = torch.tensor(region, device=device)

    cov_tr = kernfunc(ttrain_x)
    
    if diagreg == 'Sigma':
        diagreg = torch.diag(torch.tensor(train_sig2, device=device))
    else:
        diagreg = torch.diag(torch.tensor(train_sig2, device=device)) + torch.eye(len(ttrain_y))*diagreg

    cov_tr += diagreg
    if check_CN:
        print('Post Reg CondNo=', torch.linalg.cond(cov_tr))
    mean = torch.tensor(gp_mean, device=device) if gp_mean is not None else torch.mean(ttrain_y)
    tsolve_prod = torch.linalg.solve(cov_tr, torch.tensor(ttrain_y - mean, dtype=torch.float, device=device))
    cov_test_train = kernfunc(ttest_x, ttrain_x)

    gprecon = mean + torch.matmul(cov_test_train, tsolve_prod)
    if verbose:
        return gprecon.cpu().numpy(), cov_tr.cpu().numpy()
    if return_posterr:
        print("Not implemented yet")
    else:
        return gprecon.cpu().numpy()




def gaussiankernel_reconstruction(stars, region, length=5.0, scale=None):
    # probably add nancheck
    train_x = np.hstack([stars[:, 0].reshape((-1, 1)), stars[:, 1].reshape((-1, 1))])
    train_y = stars[:, 2]
    test_x = region
    train_ysig = stars[:, 3]

    avrecon = np.zeros(len(test_x))
    recon_err = np.zeros(len(test_x))
    distmat = cdist(test_x, train_x)
    for v in range(len(avrecon)):  # Can write as matrix mult and speed up (not needed yet)
        reldist = distmat[v, :]
        if scale is None:
            scale = np.sqrt(1.0 / np.sqrt(2 * np.pi * (length ** 2)))
        weights = se_kernel(reldist, length, scale)
        nanmask = ~np.isnan(weights)
        weights = weights[nanmask]

        avrecon[v] = np.sum(np.multiply(train_y[nanmask], weights)) / np.sum(weights)
        recon_err[v] = np.sum(np.multiply(train_ysig[nanmask], weights)) / np.sum(weights)
    return avrecon, recon_err


def eqkernel_reconstruction(stars, region, length=5.0, sigma=1e-2):
    train_x = np.hstack([stars[:, 0].reshape((-1, 1)), stars[:, 1].reshape((-1, 1))])
    train_y = stars[:, 2]
    test_x = region
    train_ysig = stars[:, 3]

    rho = len(train_x) / test_x.shape[0]
    wt_func = se2d_equivalentkernel(rho, sigma, length)

    avrecon = np.zeros(len(test_x))
    distmat = cdist(test_x, train_x)
    for v in range(len(avrecon)):  # Can write as matrix mult and speed up (not needed yet)
        reldist = distmat[v, :]
        weights = wt_func(reldist)
        nanmask = ~np.isnan(weights)
        weights = weights[nanmask]
        avrecon[v] = np.sum(np.multiply(train_y[nanmask], weights)) / np.sum(weights)
    return avrecon







#MAP Experiments

def get_likelihood(stars, region, Nsideresol):
    regpix = hp.vec2pix(Nsideresol, region[:, 0], region[:, 1], region[:, 2])
    Emedian, Esigma = stars[:, -2], stars[:, -1]
    starpix = hp.vec2pix(Nsideresol, stars[:, 0], stars[:, 1], stars[:, 2])
    #consider only stars INSIDE the regpix
    regmask = np.isin(starpix, regpix)

    #get lik_pix (groupby)
    dfstars = pd.DataFrame({'pix2048': starpix[regmask], 'Emedian.Esigmainv2': Emedian[regmask]*(Esigma[regmask]**(-2)), 'Esigmainv2': Esigma[regmask]**(-2)})
    dfpixwise = dfstars.groupby('pix2048').sum()
    starlik_inv2 = dfpixwise['Esigmainv2'].to_numpy()
    starlik_mean = dfpixwise['Emedian.Esigmainv2'].to_numpy()
    starlik_mean = starlik_mean / starlik_inv2
    starpix_gb = np.array(dfpixwise.index.to_list())

    #match to regionpix
    lik_mean, lik_inv2 = np.ones(len(region))*np.nan, np.ones(len(region))*np.nan
    iregpix_sorted = np.argsort(regpix)
    regpix_sorted = regpix[iregpix_sorted]
    starposns_in_regpix_sorted = regpix_sorted.searchsorted(starpix_gb) #positions of starpix in regpixsorted regpix_sorted[out] = starpix_gb
    regpix_ori_in_regpix_sorted = regpix_sorted.searchsorted(regpix) #positions of regpix in regpix_sorted regpix_sorted[out] = regpix
    lik_mean[starposns_in_regpix_sorted] = starlik_mean
    lik_inv2[starposns_in_regpix_sorted] = starlik_inv2
    #back to regpix order
    lik_mean = lik_mean[regpix_ori_in_regpix_sorted]
    lik_inv2 = lik_inv2[regpix_ori_in_regpix_sorted]
    return lik_mean, lik_inv2

def map_estimate(stars, region, prior_mean, prior_invcov, Nsideresol, get_ratio=False):
    #smooth=False, smoothing_lengthscale=np.deg2rad(5/60). Not smoothing here.
    #transform (stars, region, Nsideresol) to lik_invcov, lik_mean
    lik_mean, lik_invcov = get_likelihood(stars, region, Nsideresol)
    map_invcov = prior_invcov + lik_invcov
    map_mean = (prior_invcov*prior_mean + lik_invcov*lik_mean) / map_invcov
    pdivlik = np.ones(len(lik_mean)) 
    pdivlik[~np.isnan(map_mean)] = (prior_invcov*prior_mean)[~np.isnan(map_mean)] / (lik_invcov*lik_mean)[~np.isnan(map_mean)]
    map_mean[np.isnan(map_mean)] = prior_mean[np.isnan(map_mean)]

    assert map_mean.shape[0] == region.shape[0]
    assert map_invcov.shape[0] == region.shape[0]
    if get_ratio:
        assert len(map_invcov.shape)==1
        return map_mean, 1.0/map_invcov, pdivlik
    else:
        if len(map_invcov.shape)==1:
            return map_mean, 1.0/map_invcov
    


