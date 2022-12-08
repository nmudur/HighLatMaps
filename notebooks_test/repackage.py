import h5py
import numpy as np
import os
import sys
import healpy as hp
import joblib

sys.path.append('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code_Nresol/')
import do_recon_tilewise
import astro_cuts
import utils_circpatch
import warnings
warnings.filterwarnings('ignore', message='divide by zero encountered in log10')
warnings.filterwarnings('ignore', message='divide by zero encountered in power')
warnings.filterwarnings('ignore', message='divide by zero encountered in true_divide')
warnings.filterwarnings('ignore', message='invalid value encountered in log10')


def append_to_existing_h5file(df_inp, file_append, key_append):
    '''
    :param df_inp: Input_dataframe
    :param file_append: H5 file to be appended to
    :param key_append: Key under which the dataframe will be added
    :return:
    '''
    print(f'Save {len(df_inp)} objects into {file_append}, key:{key_append}')
    df_inp.to_hdf(file_append, key=f't{key_append}', mode='a', append=True, complevel=9, index=False,
                  data_columns=['l', 'b'])
    return

def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    v = df.values
    cols = df.columns
    types = [(cols[i], df[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z


def append_to_existing_h5file_alternate(df_inp, file_append, key_append):
    '''
    :param df_inp: Input_dataframe
    :param file_append: H5 file to be appended to
    :param key_append: Key under which the dataframe will be added
    :return:
    '''
    print(f'Save {len(df_inp)} objects into {file_append}, key:{key_append}')
    sa = df_to_sarray(df_inp)
    with h5py.File(file_append, 'a') as hf:
        hf.create_dataset(key_append, data= sa)
    return



def process_each_original_h5file(fname, fdir_append, cuts_list, Nsideparent=16):
    '''
    :param fname: Filename in Original Directory (input_fullsky_backup/north/postprocessed)
    :param fdir_append: Directory into which the files to
    :param cuts_list:
    :param Nsideparent: Nside at which to split and save files in fdir_append
    :return:
    '''
    hf = h5py.File(fname, 'r')
    keys = np.array(list(hf['photometry'].keys()))
    hf.close()
    starfilelist = [(fname, keys)]
    df = do_recon_tilewise.convert_to_dataframe(starfilelist)

    # converted to 103 col df and removed nan stars
    # make cuts, and add those columns to the dataframe so it now has 109 cols
    lold, bold = df['l'].to_numpy(copy=True), df['b'].to_numpy(copy=True)
    cutfuncs = [getattr(astro_cuts, ctup[0]) for ctup in cuts_list]
    masklist = []
    for c, ctup in enumerate(cuts_list):
        if len(ctup) == 1:
            masklist.append(cutfuncs[c](df))
        else:
            masklist.append(cutfuncs[c](df, **ctup[1]))
    for i in range(len(masklist)):
        df[f'Sel{i}'] = masklist[i]

    # since you're paranoid make sure the lb values dont change after making cuts
    assert np.array_equal(df['l'].to_numpy(), lold)
    assert np.array_equal(df['b'].to_numpy(), bold)

    # remove extra columns from the dataframe to make it a smaller file
    drop_columns = ['tmass_gal_contam', 'gaia_edr3.pmra', 'gaia_edr3.pmra_error', 'gaia_edr3.pmdec',
                    'gaia_edr3.pmdec_error',
                    'gaia_edr3.astrometric_excess_noise', 'ast_chi2', 'ast_n_good_obs', 'visibility_periods_used',
                    'posterior_mean_dm', 'posterior_mean_E', 'posterior_sigma_dm', 'posterior_sigma_E',
                    'allwise.rchi2', 'allwise.w1rchi2', 'allwise.w1rchi2_pm',
                    'allwise.w2rchi2', 'allwise.w2mpro', 'allwise.w2sigmpro', 'allwise.w3rchi2', 'allwise.w3mpro',
                    'allwise.w3sigmpro', 'allwise.w4mpro', 'allwise.w4sigmpro', 'DiscPlx',
                    'ps.psfmagmean_g', 'ps.psfmagstdev_g', 'ps.apmagmean_g', 'ps.apmagstdev_g', 'ps.psf-apmag_g',
                    'ps.psfmagmean_r', 'ps.psfmagstdev_r', 'ps.apmagmean_r', 'ps.apmagstdev_r', 'ps.psf-apmag_r',
                    'ps.psfmagmean_i', 'ps.psfmagstdev_i', 'ps.apmagmean_i', 'ps.apmagstdev_i', 'ps.psf-apmag_i',
                    'ps.psfmagmean_z', 'ps.psfmagstdev_z', 'ps.apmagmean_z', 'ps.apmagstdev_z', 'ps.psf-apmag_z',
                    'ps.psfmagmean_y', 'ps.psfmagstdev_y', 'ps.apmagmean_y', 'ps.apmagstdev_y', 'ps.psf-apmag_y']
    df = df.astype({col: 'int64' for col in df.select_dtypes('uint64').columns}) #doesnt handle uints
    df.rename(columns={"gaia_edr3.source_id": "gaia_edr3_source_id"}, inplace=True)
    df.drop(drop_columns, axis=1, inplace=True)
    
    # selection criterion (keeping excluded or included stars)
    # assuming you want to save the included objects (included because of Cuts1-5)=>all of the last 4 columns should be 1
    # don't care about the first distance cut
    save_mask = df['Sel1'].to_numpy() * df['Sel2'].to_numpy() * df['Sel3'].to_numpy() * df['Sel4'].to_numpy() * df[
        'Sel5'].to_numpy()
    dfsave = df.iloc[save_mask, :]
    print('Percentage saved', 100 * len(dfsave) / len(df))

    # filter by position on sky
    bmask = np.abs(dfsave['b'].to_numpy()) >= 29
    if np.sum(bmask) > 0:
        dfsave = dfsave.iloc[bmask, :]
        print('Percentage saved, after b>=29', 100 * len(dfsave) / len(df))
        # extract the number of the og file and make that the key in the new file
        start = fname.rindex('/')
        buff = len('north.')  # or south. (length is the same so it's fine)
        fname_key = fname[start + buff + 1:fname.rindex('.')]

        parent_pix = hp.ang2pix(Nsideparent, dfsave['l'].to_numpy(), dfsave['b'].to_numpy(), lonlat=True) # since you're using lb, no nest worry
        # parent pixels at Nsideparent (basically the files in which the filtered dataframe needs to be appended to)
        uniquepp = np.unique(parent_pix)
        if len(uniquepp) == 1:
            fappend = os.path.join(fdir_append, f'pix{Nsideparent}_{uniquepp[0]}.h5')
            append_to_existing_h5file_alternate(dfsave, file_append=fappend, key_append=fname_key) #MOD
            print(f'Appended {len(dfsave)} out of {len(dfsave)} to pix{Nsideparent}_{uniquepp[0]}')
        else:
            for ip, parent in enumerate(uniquepp):
                dfsubset = dfsave.iloc[parent_pix == parent, :]
                fappend = os.path.join(fdir_append, f'pix{Nsideparent}_{parent}.h5')
                append_to_existing_h5file_alternate(dfsubset, file_append=fappend, key_append=fname_key) #MOD
                print(f'Appended {len(dfsubset)} out of {len(dfsave)} to pix{Nsideparent}_{parent}')
    else:
        print('No objects passing cuts and |b|>=29')
    print('#############################')
    return



def process_each_original_h5file_debug(fname, fdir_append, cuts_list, Nsideparent=16):
    '''
    :param fname: Filename in Original Directory (input_fullsky_backup/north/postprocessed)
    :param fdir_append: Directory into which the files to
    :param cuts_list:
    :param Nsideparent: Nside at which to split and save files in fdir_append
    :return:
    '''
    hf = h5py.File(fname, 'r')
    keys = np.array(list(hf['photometry'].keys()))
    hf.close()
    starfilelist = [(fname, keys)]
    df = do_recon_tilewise.convert_to_dataframe(starfilelist)

    # converted to 103 col df and removed nan stars
    # make cuts, and add those columns to the dataframe so it now has 109 cols
    lold, bold = df['l'].to_numpy(copy=True), df['b'].to_numpy(copy=True)
    cutfuncs = [getattr(astro_cuts, ctup[0]) for ctup in cuts_list]
    masklist = []
    for c, ctup in enumerate(cuts_list):
        if len(ctup) == 1:
            masklist.append(cutfuncs[c](df))
        else:
            masklist.append(cutfuncs[c](df, **ctup[1]))
    for i in range(len(masklist)):
        df[f'Sel{i}'] = masklist[i]

    # since you're paranoid make sure the lb values dont change after making cuts
    assert np.array_equal(df['l'].to_numpy(), lold)
    assert np.array_equal(df['b'].to_numpy(), bold)

    # remove extra columns from the dataframe to make it a smaller file
    drop_columns = ['tmass_gal_contam', 'gaia_edr3.pmra', 'gaia_edr3.pmra_error', 'gaia_edr3.pmdec',
                    'gaia_edr3.pmdec_error',
                    'gaia_edr3.astrometric_excess_noise', 'ast_chi2', 'ast_n_good_obs', 'visibility_periods_used',
                    'posterior_mean_dm', 'posterior_mean_E', 'posterior_sigma_dm', 'posterior_sigma_E',
                    'allwise.rchi2', 'allwise.w1rchi2', 'allwise.w1rchi2_pm',
                    'allwise.w2rchi2', 'allwise.w2mpro', 'allwise.w2sigmpro', 'allwise.w3rchi2', 'allwise.w3mpro',
                    'allwise.w3sigmpro', 'allwise.w4mpro', 'allwise.w4sigmpro', 'DiscPlx',
                    'ps.psfmagmean_g', 'ps.psfmagstdev_g', 'ps.apmagmean_g', 'ps.apmagstdev_g', 'ps.psf-apmag_g',
                    'ps.psfmagmean_r', 'ps.psfmagstdev_r', 'ps.apmagmean_r', 'ps.apmagstdev_r', 'ps.psf-apmag_r',
                    'ps.psfmagmean_i', 'ps.psfmagstdev_i', 'ps.apmagmean_i', 'ps.apmagstdev_i', 'ps.psf-apmag_i',
                    'ps.psfmagmean_z', 'ps.psfmagstdev_z', 'ps.apmagmean_z', 'ps.apmagstdev_z', 'ps.psf-apmag_z',
                    'ps.psfmagmean_y', 'ps.psfmagstdev_y', 'ps.apmagmean_y', 'ps.apmagstdev_y', 'ps.psf-apmag_y']
    df = df.astype({col: 'int64' for col in df.select_dtypes('uint64').columns}) #doesnt handle uints
    df.rename(columns={"gaia_edr3.source_id": "gaia_edr3_source_id"}, inplace=True)
    df.drop(drop_columns, axis=1, inplace=True)
    
    # selection criterion (keeping excluded or included stars)
    # assuming you want to save the included objects (included because of Cuts1-5)=>all of the last 4 columns should be 1
    # don't care about the first distance cut
    save_mask = df['Sel1'].to_numpy() * df['Sel2'].to_numpy() * df['Sel3'].to_numpy() * df['Sel4'].to_numpy() * df[
        'Sel5'].to_numpy()
    dfsave = df.iloc[save_mask, :]
    print('Percentage saved', 100 * len(dfsave) / len(df))

    # filter by position on sky
    bmask = np.abs(dfsave['b'].to_numpy()) >= 29
    if np.sum(bmask) > 0:
        dfsave = dfsave.iloc[bmask, :]
        print('#############################', flush=True)
        return dfsave
    else:
        print('No objects passing cuts and |b|>=29')
        return

if __name__=='__main__':
    nmdir = '../notebooks_main1/'
    wisemodel = joblib.load(nmdir + 'spectra_matched/models/svm_panstarrs_quasarsep_balanced.joblib')
    smodel = joblib.load(nmdir + 'spectra_matched/models/svm_sdss_quasarsep_balanced_new.joblib')

    cuts_list = [('distmod_median_cut_corr', {'minpc': 400}), ('dm_sigma_cut', {'maxsig': 1.5}),
                 ('wise_svmnondetectioncombinedcut_limiting', {'model': wisemodel}), ('parallax_nan_edr3',),
                 ('sdss_uvcut', {'model': smodel}), ('bayestar_chisq', {'maxchisq': 3.0})]

    sky_part = sys.argv[1]
    assert sky_part in ['north', 'south']
    file_prefix = sky_part+'.{:05d}.h5'
    if sky_part=='north':
        parentdir = f'/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input_fullsky_backup/{sky_part}/postprocessed/' #USE THE BACKUP
    else:
        parentdir = f'/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input_fullsky_backup/{sky_part}/'
     
    allfilesindir = os.listdir(parentdir)
    allfiles = []
    for elem in allfilesindir:
        if elem.endswith('.h5'):
            allfiles.append(elem)
    
    fdir_append = f'/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/filtered_stars/{sky_part}/'
    print(len(allfiles))
    for ifn in range(len(allfiles)):
        fname = parentdir + file_prefix.format(ifn)
        print(fname)
        process_each_original_h5file(fname, fdir_append=fdir_append, cuts_list=cuts_list, Nsideparent=16)
