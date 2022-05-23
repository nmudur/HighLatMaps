import subprocess
import numpy as np
from random import randint
import sys
import astropy
import h5py
import os
import scipy
import scipy.stats

from astropy.table import Table
from astropy.io import fits


'''
output_dir = '/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/output/'
input_dir = '/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input/'

egrid = np.arange(700)*0.01
dm_grid = 4 + np.arange(120)*0.125

def get_index_percentiles_from_distribution(starr):
    
    #Array with (Nstars, E, distmod)
    #input should already have the nopost stars masked out
    
    assert starr.shape[1]==700
    assert starr.shape[2]==120
    Nstars = starr.shape[0]
    #grid
    post_Empdf = np.sum(starr, axis=-1)
    post_dmodmpdf = np.sum(starr, axis=1)
    #cdf
    post_Emcdf = np.cumsum(post_Empdf, axis=-1)
    post_dmodmcdf = np.cumsum(post_dmodmpdf, axis=-1)
    
    #find pctiles
    iEpctarray = np.empty((Nstars, 3), dtype='int')
    idmodpctarray = np.empty((Nstars, 3), dtype='int')
    
    for ip, pct in enumerate([0.16, 0.50, 0.84]):
        iEpctarray[:, ip] = np.argmin(np.abs(post_Emcdf - pct), axis=1)    
        idmodpctarray[:, ip] = np.argmin(np.abs(post_dmodmcdf - pct), axis=1)
            
    return iEpctarray, idmodpctarray


def combine_input_and_output(stripe_file):
    
    #Links the input and output files to generate <something that astropy.table can read>
    #Then use LSD make_LSD_schema to reprocess this into a bayestar19like table
    

    Numtotstars = 0
    nopoststars = 0
    
    #single stripe.00030.h5 type file
    inpf = h5py.File(input_dir+stripe_file, 'r')
    outf = h5py.File(output_dir+stripe_file, 'r')

    assert inpf['photometry'].keys() == outf.keys()
    
    pixtables = []
    for pixel in inpf['photometry'].keys():
        #print(pixel)
        #a single pixel
        inptable = astropy.io.misc.hdf5.read_table_hdf5(inpf, path='photometry/{}'.format(pixel))
        stelpost = np.array(outf[pixel]['stellar pdfs'])
        nopostmask = ~np.all(stelpost==0, axis=(1, 2)) #all grid eval pdf points are 0

        ipct_e, ipct_distmod = get_index_percentiles_from_distribution(stelpost[nopostmask, :, :])
        pct_e, pct_dm = astropy.table.Column(name='percentiles_E', data=egrid[ipct_e]), astropy.table.Column(name='percentiles_dm', data=dm_grid[ipct_distmod])
        Numtotstars+=len(stelpost)
        nopoststars+=(~nopostmask).sum()
        pixtable = inptable[nopostmask]
        pixtable.add_column(pct_e)
        pixtable.add_column(pct_dm)
        pixtables.append(pixtable)
    filetable = astropy.table.vstack(pixtables)
    print('NumStars', Numtotstars)
    print('Num invalid', nopoststars)
    inpf.close()
    outf.close()
    return filetable
'''

###Function to calculate the mean, sigma of the posterior GMM
def get_gmm_mean_sigma(f, pixel):
    '''
    f: h5py. file
    pixel: pixel id for which you're processing the file to get the mean and cov of the GMM. eg. pixel 512-602370

    Handles nans by returning nans
    '''
    t = astropy.io.misc.hdf5.read_table_hdf5(f, path='{}/gridstars'.format(pixel))
    #get mixture weights
    gmm_pi = np.array(t['ln_prior'] + t['ln_likelihood'])
    gmm_pi = np.exp(gmm_pi)
    gmm_pi = gmm_pi/np.sum(gmm_pi, axis=1).reshape((-1, 1))
    
    #get component covs
    invcov = np.array(f['{}/gridstars_icov'.format(pixel)])
    invcov00, invcov01, invcov11 = invcov[:, 0], invcov[:, 1], invcov[:, 2]
    det = invcov00*invcov11 - invcov01**2
    cov = np.vstack([invcov11, -invcov01, invcov00]).T/det.reshape((-1, 1))
    
    #get mixture means
    means = np.stack([np.array(t['dm']), np.array(t['E'])], axis=-1)
    
    #distribution mean, cov
    gmm_mean = np.sum(np.expand_dims(gmm_pi, axis=-1) * means, axis=1)
    cov_mm_00 = cov[:, 0] + np.sum(gmm_pi * (means[:, :, 0]**2), axis=1) - gmm_mean[:, 0]**2
    cov_mm_01 = cov[:, 1] + np.sum(gmm_pi * (means[:, :, 0]* means[:, :, 1]), axis=1) - (gmm_mean[:, 0]*gmm_mean[:, 1])
    cov_mm_11 = cov[:, 2] + np.sum(gmm_pi * (means[:, :, 1]**2), axis=1) - gmm_mean[:, 1]**2

    return gmm_mean, np.sqrt(np.vstack([cov_mm_00, cov_mm_11]).T)



### Functions to calculate the cdf and percentiles of the posterior GMM
def eval_marginal_gmm_cdf(x, comp_mean, comp_weight, comp_sigma):
    return np.sum(comp_weight*scipy.stats.norm.cdf(x, loc=comp_mean, scale=comp_sigma))

def compute_percentile_given_gmm_inputs_1d(comp_means, comp_weights, comp_sigmas):
    assert len(comp_means.shape)==1
    assert len(comp_sigmas.shape)==1

    #dm: calculating percentiles independently
    cdf_func = np.vectorize(lambda x: eval_marginal_gmm_cdf(x, comp_means, comp_weights, comp_sigmas))
    idx_sorted = np.argsort(comp_means) #sort by component means
    means_sorted, sigmas_sorted, weights_sorted = comp_means[idx_sorted], comp_sigmas[idx_sorted], comp_weights[idx_sorted]
    cdf_at_means = cdf_func(means_sorted)
    percs = np.array([0.1587, 0.50, 0.8413])
    idx_percs = np.searchsorted(cdf_at_means, percs, side='right')
    #print(cdf_at_means[idx_percs -1], cdf_at_means[idx_percs])
    eperc_vals, actual_percs = np.ones(3)*np.nan, np.ones(3)*np.nan
    for ip, percn in enumerate(percs):
        lower_lim = means_sorted[idx_percs[ip] -1] if idx_percs[ip]>0 else means_sorted[0]/10
        upper_lim = means_sorted[idx_percs[ip]] if idx_percs[ip]<250 else means_sorted[-1]*10
        values = np.linspace(lower_lim, upper_lim, 10)
         
        percvals = cdf_func(values)
        percdiff = np.abs(percvals - percn)
        idxmin = np.arange(len(percdiff))[np.isclose(percdiff, np.min(percdiff))][0]  #np.where(percdiff == np.min(percdiff))[0][0]
        eperc_vals[ip] = values[idxmin]
        actual_percs[ip] = percvals[idxmin]
    return eperc_vals, actual_percs

def get_gmm_percentiles(f, pixel):
    '''
    f: h5py. file
    pixel: pixel id for which you're processing the file to get the mean and cov of the GMM. eg. pixel 512-602370
    '''
    t = astropy.io.misc.hdf5.read_table_hdf5(f, path='{}/gridstars'.format(pixel))
    # get mixture weights
    gmm_pi = np.array(t['ln_prior'] + t['ln_likelihood'])
    gmm_pi = np.exp(gmm_pi)
    gmm_pi = gmm_pi / np.sum(gmm_pi, axis=1).reshape((-1, 1)) #Nstar x Ncomp
    nanmask = np.any(np.isnan(gmm_pi), axis=1)
    if nanmask.sum()>0:
        print('Pixel: {} Nanstars={}'.format(pixel, nanmask.sum()))
    
    # get component covs
    invcov = np.array(f['{}/gridstars_icov'.format(pixel)])
    invcov00, invcov01, invcov11 = invcov[:, 0], invcov[:, 1], invcov[:, 2]
    det = invcov00 * invcov11 - invcov01 ** 2
    cov = np.vstack([invcov11, -invcov01, invcov00]).T / det.reshape((-1, 1)) #Nstar x 3
    sigmas = np.sqrt(cov[:, [0, -1]])
    # get mixture means
    means = np.stack([np.array(t['dm']), np.array(t['E'])], axis=-1) #check: NstarxNcompx2

    # distribution percs
    dm_values = np.ones((gmm_pi.shape[0], 3))*(np.nan)
    e_values = np.ones((gmm_pi.shape[0], 3))*np.nan
    actual_perc_dm = np.ones((gmm_pi.shape[0], 3))*(np.nan)
    actual_perc_e = np.ones((gmm_pi.shape[0], 3))*np.nan
    
    Ncomp = gmm_pi.shape[1]
    
    for ist in range(gmm_pi.shape[0]):
        if not nanmask[ist]: #if nan dont do the pct comp
            #dm
            dm_values[ist, :], actual_perc_dm[ist, :]  = compute_percentile_given_gmm_inputs_1d(means[ist, :, 0], gmm_pi[ist, :], np.ones(Ncomp)*sigmas[ist, 0])
            #E
            e_values[ist, :], actual_perc_e[ist, :] = compute_percentile_given_gmm_inputs_1d(means[ist, :, 1], gmm_pi[ist, :], np.ones(Ncomp)*sigmas[ist, -1])
    return dm_values, e_values, actual_perc_dm, actual_perc_e



###Function that reads in a gridstars file, saves the necessary summary stats in the corresponding input file and deletes the gridstars file (optional)        
def postprocess_output(fname, fname_save, delete=False):
    '''
    Add additional mean and sigma columns to the corresponding input file
    '''
    f = h5py.File(fname, 'r')
    modf = h5py.File(fname_save, 'a')
    infkeys, outkeys = np.array(list(modf['photometry'].keys())), np.array(list(f.keys()))
    print('Output keys length:{} Input keys length:{}'.format(len(outkeys), len(infkeys)))
    extra_inp = ~np.isin(infkeys, outkeys)
    extra_out = ~np.isin(outkeys, infkeys)
    if (extra_inp).sum()>0:
        print('Extra pix in inp', infkeys[extra_inp]) 
    if (extra_out).sum()>0:
        print('Extra pix in out', outkeys[extra_out])
    for ip, pixel in enumerate(f.keys()):
        try:
            print(f'{ip}th Pixel {pixel}, Started GMM compute')
            mean, sigma = get_gmm_mean_sigma(f, pixel)
            dm_pct, e_pct, actual_perc_dm, actual_perc_e = get_gmm_percentiles(f, pixel)
            modf['photometry/{}'.format(pixel)]['chisq'] = f[pixel+'/star_chi2'][:]
            modf['photometry/{}'.format(pixel)]['posterior_mean_dm'] = mean[:, 0]
            modf['photometry/{}'.format(pixel)]['posterior_mean_E'] = mean[:, 1]
            modf['photometry/{}'.format(pixel)]['posterior_sigma_dm'] = sigma[:, 0]
            modf['photometry/{}'.format(pixel)]['posterior_sigma_E'] = sigma[:, 1]
            modf['photometry/{}'.format(pixel)]['percentiles_dm'] = dm_pct
            modf['photometry/{}'.format(pixel)]['percentiles_E'] = e_pct
            modf['photometry/{}'.format(pixel)]['actual_percentiles_dm'] = actual_perc_dm
            modf['photometry/{}'.format(pixel)]['actual_percentiles_E'] = actual_perc_e
            print(f'Pixel {pixel}, Ended GMM compute')
        except Exception as e:
            print(repr(e), fname, pixel)
    f.close()
    modf.close()
    if delete:
        os.remove(fname)
    return
'''
def main_old():
    fnames = np.array(os.listdir(output_dir))
    filemask = [out.endswith('.h5') for out in fnames]
    fnames = fnames[filemask]
    print('Combining files')
    for ifn, fname in enumerate(fnames):
        print(fname)
        try:
            filetable = combine_input_and_output(fname)
            if ifn==0:
                f = h5py.File('../output/combined_patch.h5', 'w')
                astropy.io.misc.hdf5.write_table_hdf5(filetable, f, path=fname[:fname.rindex('.')], compression=5)
                f.close()
            else: 
                f = h5py.File('../output/combined_patch.h5', 'a')
                astropy.io.misc.hdf5.write_table_hdf5(filetable, f, path=fname[:fname.rindex('.')], compression=5, append=True)
                f.close()
        except OSError:
            print('OSError for file', fname)
    return
'''

if __name__=='__main__':
    #main()
    #print(sys.argv)
    filename_gridstars, filename_final, delete_bool = sys.argv[1], sys.argv[2], True if sys.argv[3]=='delete' else False
    print(filename_gridstars, filename_final)
    
    if delete_bool: print("Deleting {}".format(filename_gridstars))
    postprocess_output(filename_gridstars, filename_final, delete_bool)
