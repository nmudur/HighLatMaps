import h5py
import os
import sys
import numpy as np

import astropy
from astropy.table import Table

def older_criterion_mask(ftest, pixel):
    return (ftest['photometry'][pixel]['visibility_periods_used']>8) * (ftest['photometry'][pixel]['gaia.ruwe']<1.4) * \
    np.isfinite(ftest['photometry'][pixel]['gaia.parallax']) * np.isfinite(ftest['photometry'][pixel]['gaia.parallax_error'])

def check_consistency(ftest, pixel):
    #Checks that the criterion for considering gaia plxs is not vastly different than just ruwe<1.4
    newmask = (ftest['photometry/{}'.format(pixel)]['pi_err']<1e6)
    oldfinerr = ftest['photometry/{}'.format(pixel)]['gaia.parallax_error']<1e6
    oldmask = older_criterion_mask(ftest, pixel) * oldfinerr
    print('Old Criterion: {:.1f}. New Criterion: {:.1f}'.format(100*np.sum(oldmask)/len(oldmask), 100*np.sum(newmask)/len(newmask)))
    disc = 100*((newmask[~oldmask]).sum()/len(newmask) + (~newmask[oldmask]).sum()/len(newmask))
    if disc>2:
        print('Disc = ', disc)
    
    if (disc>10) * ((100*np.sum(newmask)/len(newmask))<60):
        print('Odd behavior here')
    return disc, (100*np.sum(newmask)/len(newmask))

def check_pre_bstar_input_consistency(inpdir):
    #Scans all files in a directory to identify those with the maximum discrepancy between the old and new criterion
    # and those with the fewest valid parallaxes. To be run on a directory after running fidelity_north.sbatch
    allf = os.listdir(inpdir)
    fns= []
    for f in allf:
        if f.endswith('.h5'):
            fns.append(f)
    print('Number of files', len(fns))
    maxdisc = 2
    minvalid = 80
    minval_file = None
    disc_file = None
    manyinvalid = []
    for f in fns:
        fname = os.path.join(inpdir, f)
        print(f'Fn: {f}')
        ftest = h5py.File(fname, 'r')
        for ip, pixel in enumerate(ftest['photometry/'].keys()):
            print(f'{ip} id: {pixel}')
            disc, newperc = check_consistency(ftest, pixel)
            if newperc<50:
                manyinvalid.append((fname, pixel))
            if disc>maxdisc:
                maxdisc = disc
                disc_file = (f, pixel)
            if newperc<minvalid:
                minvalid = newperc
                minval_file = (f, pixel)
        ftest.close()
        print('##############')
    
    print('################DIRECTORY SUMMARY######################')
    print('Invalid pixels', len(manyinvalid))
    print(*manyinvalid, sep = "\n")
    print(f'Pixel with fewest valid parallax sources: {minval_file[0]}, {minval_file[1]}: {minvalid}')
    print(f'Maximum discrepancy between the old and new criterion: {disc_file[0]}, {disc_file[1]}: {maxdisc}')
    return

def modify_using_nofid_criterion(inpdir):
    allf = os.listdir(inpdir)
    fns = []
    for f in allf:
        if f.endswith('.h5'):
            fns.append(f)
    print('Number of files', len(fns))
    for f in fns:
        fname = os.path.join(inpdir, f)
        print(f'Fn: {f}')
        ftest = h5py.File(fname, 'a')
        for ip, pixel in enumerate(ftest['photometry/'].keys()):
            print(f'{ip} id: {pixel}')
            assert np.sum(ftest['photometry/{}'.format(pixel)]['pi_err']<1e6)==0
            idx_gaia = older_criterion_mask(ftest, pixel)
            plxnew = np.array(ftest['photometry/{}'.format(pixel)]['pi'])
            plxerrnew = np.array(ftest['photometry/{}'.format(pixel)]['pi_err'])
            plxnew[idx_gaia] = ftest['photometry/{}'.format(pixel)]['gaia.parallax'][idx_gaia]
            plxerrnew[idx_gaia] = ftest['photometry/{}'.format(pixel)]['gaia.parallax_error'][idx_gaia]
            ftest['photometry/{}'.format(pixel)]['pi'] = plxnew
            ftest['photometry/{}'.format(pixel)]['pi_err'] = plxerrnew 
            print(f'Assigned: idx_gaia perc= {np.sum(idx_gaia)/len(idx_gaia)}. Nonzero parallax= ', np.sum(plxnew!=0), np.sum(ftest['photometry/{}'.format(pixel)]['pi']!=0), np.sum(ftest['photometry/{}'.format(pixel)]['pi_err']!=1e10))
        ftest.close()
        print('##############################')
    return 


def check_files_in_dir(inpdir):
    #Check that all files in a directory can be read
    allf = os.listdir(inpdir)
    fns = []
    for f in allf:
        if f.endswith('.h5'):
            fns.append(f)
    print('Number of files', len(fns))
    for f in fns:
        fname = os.path.join(inpdir, f)
        #print(f'Fn: {f}')
        try:
            ftest = h5py.File(fname, 'r')
            ftest.close()
        except (RuntimeError, OSError) as e:
            print(f, repr(e))

    return 

def check_postprocfiles_in_dir(inpdir):
    #Find the files with the most nan percentiles after postprocessing
    #Check directory after postproc
    allf = os.listdir(inpdir)
    fns = []
    manynans = []
    for f in allf:
        if f.endswith('.h5'):
            fns.append(f)
    print('Number of files', len(fns))
    for f in fns:
        fname = os.path.join(inpdir, f)
        #print(f'Fn: {f}')
        try:
            ftest = h5py.File(fname, 'r')
            maxnan = 0
            maxnanloc = ()
            for ip, pixel in enumerate(ftest['photometry/']):
                perc_E = ftest[f'photometry/{pixel}']['percentiles_E']
                nanperc = np.any(np.isnan(perc_E), axis=1).sum(axis=0)/perc_E.shape[0]
                if nanperc>maxnan:
                    maxnan = nanperc
                    maxnanloc = (f, pixel)
                if nanperc>0.5:
                    manynans.append((f, pixel))
            ftest.close()
            print(f'File {maxnanloc}, nanperc={100*maxnan}')
        except (RuntimeError, OSError) as e:
            print('Corrupted?', f, repr(e))
        print('##############')
    
    print('################DIRECTORY SUMMARY######################')
    print('Pixels where more than 50% stars have nan percentiles', len(manynans))
    print(*manynans, sep = "\n")
    print(f'Pixel with the most nan percentiles: {maxnanloc[0]}, {maxnanloc[1]}: {maxnan}')
    return 

def retrieve_discrepant_posteriors(fidelityfile, nofidelityfile):
    tablist = []
    for ip, pixel in enumerate(fidelityfile['photometry/'].keys()):
        fid_pass = (fidelityfile[f'photometry/{pixel}']['pi_err']<1e6)
        nofid_pass = (nofidelityfile[f'photometry/{pixel}']['pi_err']<1e6)
        discrepant = (fid_pass*(~nofid_pass)) + (nofid_pass * (~fid_pass))
        difftab = Table(fidelityfile[f'photometry/{pixel}'][discrepant])
        difftab.rename_columns(['posterior_mean_E', 'posterior_mean_dm', 'posterior_sigma_E', 'posterior_sigma_dm', 'percentiles_dm', 'percentiles_E', 'actual_percentiles_dm','actual_percentiles_E', 'chisq'],
                ['posterior_mean_E_fid', 'posterior_mean_dm_fid', 'posterior_sigma_E_fid', 'posterior_sigma_dm_fid', 'percentiles_dm_fid', 'percentiles_E_fid', 'actual_percentiles_dm_fid','actual_percentiles_E_fid', 'chisq_fid'])
        difftab['fid_pass'] = fid_pass[discrepant]
        difftab['nofid_pass'] = nofid_pass[discrepant]
        difftab['posterior_mean_E_nofid']= nofidelityfile[f'photometry/{pixel}']['posterior_mean_E'][discrepant]
        difftab['posterior_mean_dm_nofid']= nofidelityfile[f'photometry/{pixel}']['posterior_mean_dm'][discrepant]
        difftab['posterior_sigma_E_nofid']= nofidelityfile[f'photometry/{pixel}']['posterior_sigma_E'][discrepant]
        difftab['posterior_sigma_dm_nofid']= nofidelityfile[f'photometry/{pixel}']['posterior_sigma_dm'][discrepant]
        difftab['percentiles_E_nofid']= nofidelityfile[f'photometry/{pixel}']['percentiles_E'][discrepant]
        difftab['percentiles_dm_nofid']= nofidelityfile[f'photometry/{pixel}']['percentiles_dm'][discrepant]
        difftab['actual_percentiles_E_nofid']= nofidelityfile[f'photometry/{pixel}']['actual_percentiles_E'][discrepant]
        difftab['actual_percentiles_dm_nofid']= nofidelityfile[f'photometry/{pixel}']['actual_percentiles_dm'][discrepant]
        tablist.append(difftab)
    disctable = astropy.table.vstack(tablist)
    return disctable

def get_all_stars_in_file(fname):
    tablist = []
    for ip, pixel in enumerate(fname['photometry/'].keys()):
        tablist.append(Table(fname[f'photometry/{pixel}'][:]))
    return astropy.table.vstack(tablist)

if __name__=='__main__':
    '''
    #Checks the difference between the old and new criteria for files which have the fidelity criteria saved 
    INPDIR = '/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input_fullsky/north/'
    check_pre_bstar_input_consistency(INPDIR)
    '''
    #Postprocessed file check
    INPDIR = '/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input_fullsky/north/postprocessed/'
    check_postprocfiles_in_dir(INPDIR)

    '''
    #Modifies a bunch of input files so as to set the plx fields according to the old criteria
    INPDIR = '/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input_PoleNoFid/'
    modify_using_nofid_criterion(INPDIR)
    '''
