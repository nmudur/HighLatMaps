import numpy as np
import pandas as pd
import sys
import os
import h5py
import astropy
import matplotlib.pyplot as plt

import astro_cuts

from astropy.io import fits
from astropy.table import Table
from sklearn.model_selection import train_test_split
sys.path.append('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code_Nresol/')
import do_recon_tilewise



#convert from fname to df
def convert_to_dataframe_specmatched(starfile):
    #adds all post-proc columns here
    dat = Table.read(starfile, format='fits')  # for superchunk
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    df['dm_median'] = dat['percentiles_dm'][:, 1]
    df['E_median'] = dat['percentiles_E'][:, 1]
    df['dm_sigma'] = (dat['percentiles_dm'][:, 2] - dat['percentiles_dm'][:, 0]) / 2
    df['E_sigma'] = (dat['percentiles_E'][:, 2] - dat['percentiles_E'][:, 0]) / 2
    df['DiscPlx'] = np.logical_xor((np.isnan(df['plx'].to_numpy())),
                                   (np.isnan(df['gaia_dr2_source.parallax'].to_numpy())))
    
    for ib, b in enumerate(['g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']):
        df['mag_'+b] = np.array(dat['mag'][:, ib])
        df['mag_err_' + b] = np.array(dat['mag_err'][:, ib])
    df['g-r'] = df['mag_g'].to_numpy() - df['mag_r'].to_numpy()
    df['r-i'] = df['mag_r'].to_numpy() - df['mag_i'].to_numpy()
    df['z-W1'] = df['mag_z'].to_numpy() - df['allwise.w1mpro'].to_numpy()
    df['W1-W2'] = df['allwise.w1mpro'].to_numpy() - df['allwise.w2mpro'].to_numpy()
    df['W2-W3'] = df['allwise.w2mpro'].to_numpy() - df['allwise.w3mpro'].to_numpy()
    n_passbands = np.count_nonzero(np.isfinite(dat['mag_err']), axis=1)
    df['reduced_chisq'] = df['chisq'].to_numpy() * n_passbands / (n_passbands - 4)
    
    
    if len([c.startswith('sdss_dr14_starsweep') for c in dat.colnames])>0:
        sdss_flux_sig = np.power(np.array(dat['sdss_dr14_starsweep.psfflux_ivar']), -0.5)
        for ib, b in enumerate(['u', 'g', 'r', 'i', 'z']):
            df['sdss.pmag_'+b] = 22.5 - 2.5*np.clip(np.log10(np.array(dat['sdss_dr14_starsweep.psfflux'])[:, ib]), 0.0, np.inf)
            df['sdss.pmag_err_'+b] = (2.5/np.log(10)) * (sdss_flux_sig[:, ib]/np.array(dat['sdss_dr14_starsweep.psfflux'])[:, ib])
        df['u-g_sdss'] = df['sdss.pmag_u'].to_numpy() - df['sdss.pmag_g'].to_numpy()
        df['g-r_sdss'] = df['sdss.pmag_g'].to_numpy() - df['sdss.pmag_r'].to_numpy()
    
    if 'sdss_dr17_specobj.SN_MEDIAN' in dat.colnames:
        for ib, b in enumerate(['u', 'g', 'r', 'i', 'z']):
            df['sn_median_{}'.format(b)] = dat['sdss_dr17_specobj.SN_MEDIAN'][:, ib]
    df['label'] = np.array(~np.logical_or(np.array(df['sdss_dr17_specobj.CLASS']==b'QSO'), 
                                         np.array(df['sdss_dr17_specobj.CLASS']==b'GALAXY')), dtype='int')

    return df

def convert_to_dataframe_specmatched_h5(starfile):
    #starfile: is a list of (file, pixels)
    #Omitted the sdss.sn_median conversion
    #df = do_recon_tilewise.convert_h5tilemapper_to_dataframe(starfile)
    #convert_h5tilemapper_to_dataframe modified for here
    dflist = []

    for elem in starfile:
        inputfile = h5py.File(elem[0], 'r')
        pixels_all = elem[1]  # inputfile['photometry'].keys()

        for pixel in pixels_all:
            dat = astropy.io.misc.hdf5.read_table_hdf5(inputfile, path='photometry/{}'.format(pixel))

            names = [name for name in dat.colnames if len(dat[name].shape) <= 1]  # single columns
            df = dat[names].to_pandas()
            
            rencols = {"pi": "plx", "pi_err": "plx_err"}
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
            if 'sdss_dr17_specobj.SN_MEDIAN' in dat.colnames:
                for ib, b in enumerate(['u', 'g', 'r', 'i', 'z']):
                    df['sn_median_{}'.format(b)] = dat['sdss_dr17_specobj.SN_MEDIAN'][:, ib]
    df = pd.concat(dflist)
    print('Stars Pre Cuts in h5mapper2df', len(df))
    df['label'] = np.array(~np.logical_or(np.array(df['sdss_dr17_specobj.CLASS']==b'QSO'), 
                                         np.array(df['sdss_dr17_specobj.CLASS']==b'GALAXY')), dtype='int')
    return df

#quality cuts on sdss spectra
def return_goodspec(df_input):
    cut1 = (df_input['sdss_dr17_specobj.RCHI2'].to_numpy() < 2)
    cut2 = (df_input['sdss_dr17_specobj.CHI68P'].to_numpy() <2)
    cut3 = (df_input['sdss_dr17_specobj.SN_MEDIAN_ALL'].to_numpy()>10)
    effcut = cut1*cut2*cut3
    print('Number of objects before / after', len(df_input), np.sum(effcut))
    print('c1 = {:.3f}, c2 = {:.3f}, c3 = {:.3f}'.format(np.sum(cut1)/len(cut1), np.sum(cut2)/len(cut2), np.sum(cut3)/len(cut3)))
    print('Fraction left = {:.3f}'.format(np.sum(effcut)/len(effcut)))
    return df_input.iloc[effcut, :]

#print distribution
def print_obj_distribution(classes):
    starmask, galmask, qmask = (classes==b'STAR'), (classes==b'GALAXY'), (classes==b'QSO')
    print('Number of Stars = {}, Fraction(%) = {:.4f}, Ratio(obj:star)*100 = {}'.format(np.sum(starmask), 100*np.sum(starmask)/len(starmask), 100))
    print('Number of Galaxies = {}, Fraction(%) = {:.4f}, Ratio(obj:star)*100 = {:.4f}'.format(np.sum(galmask), 100*np.sum(galmask)/len(galmask), 100*np.sum(galmask)/np.sum(starmask)))
    print('Number of QSOs = {}, Fraction(%) = {:.4f}, Ratio(obj:star)*100 = {:.4f}'.format(np.sum(qmask), 100*np.sum(qmask)/len(qmask), 100*np.sum(qmask)/np.sum(starmask)))
    return



#print delta distribution
def print_delta_obj_distribution(classes_before, classes_after):
    #before
    starmask_before, galmask_before, qmask_before = (classes_before==b'STAR'), (classes_before==b'GALAXY'), (classes_before==b'QSO')
    stfrac_before, galfrac_before, qfrac_before = np.sum(starmask_before)/len(starmask_before), np.sum(galmask_before)/len(galmask_before), np.sum(qmask_before)/len(qmask_before)
    #after
    starmask_after, galmask_after, qmask_after = (classes_after==b'STAR'), (classes_after==b'GALAXY'), (classes_after==b'QSO')
    stfrac_after, galfrac_after, qfrac_after = np.sum(starmask_after)/len(starmask_after), np.sum(galmask_after)/len(galmask_after), np.sum(qmask_after)/len(qmask_after)
    
    #fracchange
    print('StarFracChange = {:.3f} (%)'.format(100*(stfrac_after - stfrac_before)))
    print('GalFracChange = {:.3f} (%)'.format(100*(galfrac_after - galfrac_before)))
    print('QSOFracChange = {:.3f} (%)'.format(100*(qfrac_after - qfrac_before)))
    print('Stars thrown out =', np.sum(starmask_after) - np.sum(starmask_before))
    
    return


#return split
def return_train_test_subset(df_input, features, balance_train=False):
    qmask = (df_input['label'].to_numpy() == 0)
    idx_all = np.arange(len(df_input))
    #select indices
    qtr, qte = train_test_split(idx_all[qmask], test_size=0.3, random_state=24, shuffle=True)
    s_tr, s_te = train_test_split(idx_all[~qmask], test_size=0.3, random_state=24, shuffle=True)
    
    if balance_train:
        assert len(qtr)<len(s_tr)
        s_tr = s_tr[:len(qtr)] #only select as many stars as nonstars in the train
        #demand |test| = 0.3|train|
        numtrain = len(qtr) + len(s_tr)
        numtest = int(0.3 * numtrain)
        rng = np.random.default_rng(seed=24)
        test_subset = rng.choice(len(qte) + len(s_te), numtest, replace=False)
        trainidx, testidx = np.hstack([qtr, s_tr]), np.hstack([qte, s_te])[test_subset]
    else:#don't balance
        trainidx, testidx = np.hstack([qtr, s_tr]), np.hstack([qte, s_te])
    
    train_x = df_input[features].iloc[trainidx, :].to_numpy()
    test_x = df_input[features].iloc[testidx, :].to_numpy()
    train_y = df_input['label'].iloc[trainidx].to_numpy()
    test_y = df_input['label'].iloc[testidx].to_numpy()
    print('NumTrain, NumTest = ', len(trainidx), len(testidx))
    print('Train Class Fraction: Q = {}'.format(np.sum(train_y==0)/len(train_y)))
    print('Test Class Fraction: Q = {}'.format(np.sum(test_y==0)/len(test_y)))

    return train_x, train_y, test_x, test_y


#For rerun
def get_pix_to_filemapper(inputdir, dirfiles, pix256n):
    #returns a dict with pixel: file, where the pixels are all pixels that
    #specmatched sources belong to
    pix2file = {}
    
    # trying not to open files |tiles_recon|*|num_files| times so 2 loops
    for dirfile in dirfiles: #Loop over files
        inputfile = h5py.File(inputdir + dirfile, 'r')
        keys_all = np.array(list(inputfile['photometry'].keys()))
        Nsidepix, pixels_nested = np.array(
            [int(key[len('pixel '):key.rindex('-')]) for key in keys_all]), \
                                  np.array([int(key.split('-')[1]) for key in keys_all]) #[512...512], [pix_id@Nside=512]
        
        assert len(np.unique(Nsidepix))==1
        assert Nsidepix[0]==256
        match = np.isin(pix256n, pixels_nested)
        for pixel in pix256n[match]:#all specmatched pixels contained in dirfile
            if pixel in pix2file.keys():
                pix2file[pixel].append(dirfile)
            else:
                pix2file[pixel] = [dirfile]
        inputfile.close()
    return pix2file

def invert_dict(dictmapper):
    #returns file: specmatched pixels
    invmap = {}
    for k, vlist in dictmapper.items():
        for v in vlist:
            if v in invmap.keys():
                invmap[v].append(k)
            else:
                invmap[v] = [k]
    return invmap


