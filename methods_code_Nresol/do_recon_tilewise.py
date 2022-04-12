import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import sys
import psutil
import h5py
import pickle
import astropy
import time
import functools
import healpy as hp

from multiprocessing import Pool
from astropy.table import Table
from functools import reduce

import methods_cluster
from methods_cluster import *
import astro_cuts
from utils_circpatch import *



def map_h5files_to_tiles_proper(inputdir, tiles_recon, Nsidetile=16):
    '''
    :param inputdir: Directory with postprocessed input files
    :param Nsidetile: The Nside corresponding to the pixels in tiles_recon
    This looks up a directory with bayestar outputs and retrieves a list of relevant h5 files and keys for each pixel
    :return: # dictionary with tiles @ Nsidetile as key: List of [hdf5 file, list of key strings within BUFFER_RAD of tile]
    '''

    dirfiles = np.array(os.listdir(inputdir))
    dirfiles = dirfiles[np.array([dirn.endswith('.h5') for dirn in dirfiles])]
    tiledict = {}
    
    # trying not to open files |tiles_recon|*|num_files| times so 2 loops
    for dirfile in dirfiles: #Loop over files
        inputfile = h5py.File(inputdir + dirfile, 'r')
        keys_all = np.array(list(inputfile['photometry'].keys()))
        print(len(keys_all))
        Nsidepix, pixels_nested = np.array(
            [int(key[len('pixel '):key.rindex('-')]) for key in keys_all]), \
                                  np.array([int(key.split('-')[1]) for key in keys_all]) #[512...512], [pix_id@Nside=512]
        pixels = hp.nest2ring(Nsidepix, pixels_nested)
        if len(np.unique(Nsidepix)) > 1:
            raise NotImplementedError
        Nsidepix = Nsidepix[0] #always 512
        try:
            starvec = np.vstack(hp.pixelfunc.pix2vec(Nsidepix, pixels)).T  # Nstarx3, Pixkey centers

            for it, tile in enumerate(tiles_recon):  # find all relevant keys to each tile that we're reconstructing for
                #print('tile', tile)
                pixcenvec = np.array(hp.pixelfunc.pix2vec(Nsidetile, tile))
                RADIUS = 4.8
                max_radius_cosine = np.cos(np.deg2rad(RADIUS))  # furthest distance of relevant hdf5 pixels from tile
                regmask = (np.matmul(starvec, pixcenvec.reshape((3, 1)))) > max_radius_cosine
                # all starpix with their centers within RADIUS of the tile center must remain
                regmask = regmask.flatten()
                regkeys = keys_all[regmask]
                #(file_dest, relevant_keys_list)
                if regmask.sum()>0:
                    filetup = (inputdir + dirfile, regkeys)
                    if tile in tiledict:
                        tiledict[tile] = tiledict[tile] + [filetup]
                    else:
                        tiledict.update({tile: [filetup]})
                    
            inputfile.close()
        except FloatingPointError as r:
            print(dirfile, Nsidetile, tile, pixcenvec)
    return tiledict


def convert_h5tilemapper_to_dataframe(starfile):
    '''
    :param starfile: list
    Does convert_to_dataframe for the bayestar reruns
    :return:
    '''
    USE_PERCENTILES= False #first stripe rerun had USE_PERCENTILES=False

    dflist = []

    for elem in starfile:
        inputfile = h5py.File(elem[0], 'r')
        pixels_all = elem[1]  # inputfile['photometry'].keys()

        for pixel in pixels_all:
            dat = astropy.io.misc.hdf5.read_table_hdf5(inputfile, path='photometry/{}'.format(pixel))

            names = [name for name in dat.colnames if len(dat[name].shape) <= 1]  # single columns
            df = dat[names].to_pandas()
            # rename columns: using posterior mean here
            if USE_PERCENTILES:
                rencols = {"pi": "plx", "pi_err": "plx_err"}
                nanmask = np.any(np.isnan(dat['percentiles_E']), axis=1) + np.any(np.isnan(dat['percentiles_dm']), axis=1)
                df['dm_median'] = dat['percentiles_dm'][:, 1]
                df['E_median'] = dat['percentiles_E'][:, 1]
                df['dm_sigma'] = (dat['percentiles_dm'][:, 2] - dat['percentiles_dm'][:, 0]) / 2
                df['E_sigma'] = (dat['percentiles_E'][:, 2] - dat['percentiles_E'][:, 0]) / 2

            else: #use mean and sigma of the GMM
                rencols = {"posterior_mean_dm": "dm_median", "posterior_mean_E": "E_median",
                           "posterior_sigma_dm": "dm_sigma",
                           "posterior_sigma_E": "E_sigma", "pi": "plx", "pi_err": "plx_err"}
                nanmask = np.isnan(dat['posterior_mean_E'])
            colnames = np.array(dat.colnames)
            gaiacols = colnames[np.array([True if name.startswith('gaia.') else False for name in dat.colnames])]
            rengaia = {gcol: 'gaia_edr3.' + gcol[5:] for gcol in gaiacols}
            # rename gaia columns
            rencols.update(rengaia)
            df.rename(columns=rencols, inplace=True)
            df['DiscPlx'] = np.logical_xor((np.isnan(df['plx'].to_numpy())),
                                           (np.isnan(df['gaia_edr3.parallax'].to_numpy())))

            if 'err' in dat.colnames:
                for ib, b in enumerate(['g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']):
                    df['mag_' + b] = np.array(dat['mag'][:, ib])
                    df['mag_err_' + b] = np.array(dat['err'][:, ib])
                    if ib < 5:
                        df['ps.psfmagmean_' + b] = np.array(dat['mag'][:, ib])
                        df['ps.psfmagstdev_' + b] = np.array(dat['err'][:, ib])
                        df['ps.apmagmean_' + b] = np.array(dat['mean_ap'][:, ib])
                        df['ps.apmagstdev_' + b] = np.array(dat['err_ap'][:, ib])
                        df['ps.psf-apmag_' + b] = df['ps.psfmagmean_' + b].to_numpy() - df[
                            'ps.apmagmean_' + b].to_numpy()
                df['g-r'] = df['mag_g'].to_numpy() - df['mag_r'].to_numpy()
                df['r-i'] = df['mag_r'].to_numpy() - df['mag_i'].to_numpy()
                if len([c.startswith('allwise') for c in dat.colnames]) > 0:
                    df['z-W1'] = df['mag_z'].to_numpy() - df['allwise.w1mpro'].to_numpy()
                n_passbands = np.count_nonzero(np.isfinite(dat['err']), axis=1)
                df['reduced_chisq'] = df['chisq'].to_numpy() * n_passbands / (n_passbands - 4)

            # if sdss
            if np.sum([c.startswith('sdss_dr14_starsweep') for c in dat.colnames]) > 0:
                sdss_flux_sig = np.power(np.array(dat['sdss_dr14_starsweep.psfflux_ivar']), -0.5)
                for ib, b in enumerate(['u', 'g', 'r', 'i', 'z']):
                    df['sdss.pmag_' + b] = 22.5 - 2.5 * np.clip(
                        np.log10(np.array(dat['sdss_dr14_starsweep.psfflux'])[:, ib]), 0.0, np.inf)
                    df['sdss.pmag_err_' + b] = (2.5 / np.log(10)) * (
                            sdss_flux_sig[:, ib] / np.array(dat['sdss_dr14_starsweep.psfflux'])[:, ib])
            dflist.append(df.iloc[~nanmask, :]) #add for each pixel
            #print(len(df.iloc[nanmask, :]))
    df = pd.concat(dflist)
    print('Stars Pre Cuts in h5mapper2df', len(df))
    return df

def convert_to_dataframe(starfile):
    # converts an input starfile to a dataframe
    # adds all post-proc columns here
    if isinstance(starfile, list):
        # List of (inputfile, relevant keys)
        print('Rerun bayestar outputs')
        df = convert_h5tilemapper_to_dataframe(starfile)

    #Below is the code that generated the main 6.1 and 15 am maps
    else:
        dat = Table.read(starfile, format='fits')  # for superchunk
        names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
        df = dat[names].to_pandas()
        df['dm_median'] = dat['percentiles_dm'][:, 1]
        df['E_median'] = dat['percentiles_E'][:, 1]
        df['dm_sigma'] = (dat['percentiles_dm'][:, 2] - dat['percentiles_dm'][:, 0]) / 2
        df['E_sigma'] = (dat['percentiles_E'][:, 2] - dat['percentiles_E'][:, 0]) / 2
        df['DiscPlx'] = np.logical_xor((np.isnan(df['plx'].to_numpy())),
                                       (np.isnan(df['gaia_dr2_source.parallax'].to_numpy())))
        if 'mag_err' in dat.colnames:
            for ib, b in enumerate(['g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']):
                df['mag_' + b] = np.array(dat['mag'][:, ib])
                df['mag_err_' + b] = np.array(dat['mag_err'][:, ib])
            df['g-r'] = df['mag_g'].to_numpy() - df['mag_r'].to_numpy()
            df['r-i'] = df['mag_r'].to_numpy() - df['mag_i'].to_numpy()
            if len([c.startswith('allwise') for c in dat.colnames]) > 0:
                df['z-W1'] = df['mag_z'].to_numpy() - df['allwise.w1mpro'].to_numpy()
            n_passbands = np.count_nonzero(np.isfinite(dat['mag_err']), axis=1)
            df['reduced_chisq'] = df['chisq'].to_numpy() * n_passbands / (n_passbands - 4)

        # if sdss
        if np.sum([c.startswith('sdss_dr14_starsweep') for c in dat.colnames]) > 0:
            sdss_flux_sig = np.power(np.array(dat['sdss_dr14_starsweep.psfflux_ivar']), -0.5)
            for ib, b in enumerate(['u', 'g', 'r', 'i', 'z']):
                df['sdss.pmag_' + b] = 22.5 - 2.5 * np.clip(
                    np.log10(np.array(dat['sdss_dr14_starsweep.psfflux'])[:, ib]), 0.0, np.inf)
                df['sdss.pmag_err_' + b] = (2.5 / np.log(10)) * (
                            sdss_flux_sig[:, ib] / np.array(dat['sdss_dr14_starsweep.psfflux'])[:, ib])
        # if ucal
        if np.sum([c.startswith('ucal_fluxqz') for c in dat.colnames]) > 0:
            for ib, b in enumerate(['g', 'r', 'i', 'z', 'y']):
                df['ps.psfmagmean_' + b] = -2.5 * np.log10(np.array(dat['ucal_fluxqz.mean'][:, ib]))
                df['ps.psfmagstdev_' + b] = (2.5 / np.log(10)) * (
                            np.array(dat['ucal_fluxqz.stdev'][:, ib]) / np.array(dat['ucal_fluxqz.mean'][:, ib]))
                df['ps.apmagmean_' + b] = -2.5 * np.log10(np.array(dat['ucal_fluxqz.mean_ap'][:, ib]))
                df['ps.apmagstdev_' + b] = (2.5 / np.log(10)) * (
                            np.array(dat['ucal_fluxqz.stdev_ap'][:, ib]) / np.array(dat['ucal_fluxqz.mean_ap'][:, ib]))
                df['ps.psf-apmag_' + b] = df['ps.psfmagmean_' + b].to_numpy() - df['ps.apmagmean_' + b].to_numpy()
    return df



def cuts_wrapper(starfile, nsidetile, tile, radius_deg_extra, cuts_list, coordstype='UnitVector', return_mask=False):
    df = convert_to_dataframe(starfile)
    print('Stars Pre Cuts in cuts_wrapper', len(df))
    # tile region cut: select stars relevant to the region you're looking at
    #All stars within a radius of <max distance from Nsidetile pixel center to edge of Nsidetile pixel + radius_deg_extra> are selected
    
    pixcenvec = np.array(hp.pixelfunc.pix2vec(nsidetile, tile))
    corners = hp.boundaries(nsidetile, tile, step=1).T
    max_radius_cosine = np.cos(
        np.max(np.arccos(np.matmul(pixcenvec.reshape((1, 3)), corners.T))) + np.deg2rad(radius_deg_extra)) # min cosine is max angsep
    starvec = hp.pixelfunc.ang2vec(df['l'].to_numpy(), df['b'].to_numpy(), lonlat=True)  # Nstarx3, check?
    regcut = (np.matmul(starvec, pixcenvec.reshape((3, 1)))) > max_radius_cosine  # cosine of pixels in the region should be greater than min cosine
    df = df.iloc[regcut, :] #df now has only the stars centered around the Nsidetile pixel_id=tile

    #get astro cuts
    cutfuncs = [getattr(astro_cuts, ctup[0]) for ctup in cuts_list]
    masklist = []
    for c, ctup in enumerate(cuts_list):
        if len(ctup)==1:
            masklist.append(cutfuncs[c](df))
        else:
            masklist.append(cutfuncs[c](df, **ctup[1]))
        print('Cut{} {} {}'.format(c, np.sum(masklist[c]), np.sum(masklist[c])/len(masklist[c])))
    final_cut = reduce(np.logical_and, masklist) #intersection of all cuts
    dfpatch = df.iloc[final_cut, :]
    print('Cuts_All {} {}'.format(np.sum(final_cut), np.sum(final_cut)/len(final_cut)))
    if coordstype=='UnitVector':
        starvec = hp.pixelfunc.ang2vec(dfpatch['l'].to_numpy(), dfpatch['b'].to_numpy(), lonlat=True)
        starspostcuts = np.hstack(
            [starvec, dfpatch['E_median'].to_numpy().reshape((-1, 1)), dfpatch['E_sigma'].to_numpy().reshape((-1, 1))])
    else:
        raise NotImplementedError #in case you wanna add 2d projected at some point? (preferably do so in the recon func)
    if return_mask:
        return starspostcuts, [regcut, final_cut] 
    #N_relevantstars x 5: columns 0-2: unit vector coordinates. column3: EMedian. column4: Esigma.
    else:
        return starspostcuts



def set_sigma_ref(star_sigmas, low=1.0, high=99.0, factor=10):
    sig2_low, sig2_high = np.percentile(star_sigmas, low)**2, np.percentile(star_sigmas, high)**2
    #1 / (sig2low + sigref2)  = f / (sig2high + sigref2)
    sigma_ref = np.sqrt((sig2_high - (factor*sig2_low))/(factor -1))
    print('SigRef', sigma_ref,  end='\n', flush = True)
    return sigma_ref


def get_data_for_tile(tile, Nsidetile, radius_deg_extra, cuts_list, Nsideresol, presaved): #edited for presaved
    '''
    tile: pixel id at Nsidetile
    Larger pixel resolution
    '''
    region = get_pixelvec_for_tile(Nsidetile, tile, Nsideresol=Nsideresol) #[Npixresol pixels in tile] x 3
    
    print('Region for tile{}'.format(tile))
    if isinstance(presaved, dict):
        #rerun format
        starfilelist = presaved[tile]
        stars = cuts_wrapper(starfilelist, nsidetile=Nsidetile, tile=tile, radius_deg_extra=radius_deg_extra, cuts_list=cuts_list, coordstype='UnitVector', return_mask=False)
        #stars = np.vstack(stars_hf) stacking now done in convert_to_dataframe

    else:
        #stars
        assert Nsidetile>=16
        startile = get_largepix_for_smallpix(tile, Nsidetile, 16) #get corresponding LSD presaved file
        starfile = presaved.format(startile)
        stars = cuts_wrapper(starfile, nsidetile=Nsidetile, tile = tile, radius_deg_extra=radius_deg_extra, cuts_list=cuts_list, coordstype='UnitVector', return_mask=False) #basically stuff that acts on a single star file
    print('Tile {}, NumPixels={}, NumStars={}, Star/Pix ratio = {}'.format(tile, region.shape[0], stars.shape[0], stars.shape[0]/region.shape[0]), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return stars, region


def get_recon_for_tile(tile, Nsidetile, radius_deg_extra, recon_func, recon_kwargs, cuts_list, Nsideresol, save=True, savdirname=None, return_vec=False, presaved='/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/data/lsdraw/stars_edr3/{}.fits'):
    #Modified default presaved to now use eDR3
    stars, region = get_data_for_tile(tile, Nsidetile, radius_deg_extra, cuts_list, Nsideresol, presaved=presaved)
    sigma_ref = set_sigma_ref(stars[:, -1])
    #setting sigma ref on the basis of the cleaned, relevant star sample for the entire Nsidetile pixel
    recon, var = recon_func(stars, region, sigref = sigma_ref, **recon_kwargs)
    if save:
        with h5py.File(savdirname+"recon_"+str(tile)+".hdf5", "w") as f:
            dset = f.create_dataset("recon", data = recon)
            dset = f.create_dataset("variance", data=var)
            dset = f.create_dataset("vec", data=region)
    if return_vec:
        return recon, var, region
    else:
        return recon, var



if __name__=='__main__':
    #Unit testing the cuts
    '''
    # can you access funcs by name?
    cut1 = getattr(astro_cuts, 'e_nonneg_cut')
    print(type(cut1))
    # yup
    # does cuts_wrapper work as before?
    # check using old cluster code
    fname = 'superchunk_r20.fits'
    make_cuts_without_saving_highlat(fname, 32, 0, radius_deg_extra=25 / 60, glue=False)
    # check using new cluster code
    cuts_list = [('distmod_median_cut', {'mindm': 8}),
                 ('e_nonneg_cut',),
                 ('e_sigma_cut', {'maxsig': 0.3}),
                 ('dm_sigma_cut', {'maxsig': 1.0}),
                 ('rel_dm_cut', {'maxerr': 1.0}),
                 ('parallax_nan',),
                 ('highlat_cut',)]
    cuts_wrapper(fname, 32, 0, radius_deg_extra=25 / 60, cuts_list=cuts_list, return_mask=True)
    '''


    #Code for when run using GNUParallel
    #argv: picklename, tile, tmpdir
    tile, tmpdir = int(sys.argv[1]), sys.argv[2]
    if not os.path.exists(tmpdir+'recon_{}.hdf5'.format(tile)):
        recon_info = pickle.load(open(tmpdir + 'recon_info.pkl', 'rb'))
        Nsidetile, radius_deg_extra, recon_kwargs, cuts_list, Nsideresol = recon_info['Nsidetile'], recon_info['radius_deg_extra'], recon_info['recon_kwargs'], recon_info['cuts_list'], recon_info['Nsideresol']
        recon_func = getattr(methods_cluster, recon_info['recon_func_name'])
        if 'stars_presaved' in recon_info.keys():
            if isinstance(recon_info['stars_presaved'], str):
                print('Using dir ', recon_info['stars_presaved'])
            out = get_recon_for_tile(tile, Nsidetile=Nsidetile, radius_deg_extra=radius_deg_extra, recon_func=recon_func, recon_kwargs=recon_kwargs, cuts_list=cuts_list, Nsideresol=Nsideresol, save=True, savdirname=tmpdir, presaved=recon_info['stars_presaved'])
        else:
            #runs with inner gaia crossmatch starfiles
            out = get_recon_for_tile(tile, Nsidetile=Nsidetile, radius_deg_extra=radius_deg_extra, recon_func=recon_func, recon_kwargs=recon_kwargs, cuts_list=cuts_list, Nsideresol=Nsideresol, save=True, savdirname=tmpdir)

