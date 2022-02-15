import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py

from astropy.coordinates import SkyCoord
import astropy.units as units

from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery
from dustmaps.planck import PlanckGNILCQuery

sys.path.append('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code/')
from utils_circpatch import *  # check if import works

sfd = SFDQuery()
gnilc = PlanckGNILCQuery()


def get_bayestar2017_map():
    with h5py.File('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/reference_maps/bayestar2017.hdf5', 'r') as f:
        b17mean = np.array(f['mean'])
        coords_lb = np.array(f['coords_lb'])
    pix2048 = hp.pixelfunc.ang2pix(2048, coords_lb[:, 0], coords_lb[:, 1], lonlat=True)
    b17map = np.empty(hp.pixelfunc.nside2npix(2048))
    b17map[pix2048] = b17mean
    return b17map

def get_sfd_map():
    pix2048 = np.arange(hp.pixelfunc.nside2npix(2048))
    selpixang = hp.pixelfunc.pix2ang(2048, pix2048, lonlat=True)
    coords = SkyCoord(l=selpixang[0] * units.deg, b=selpixang[1] * units.deg, frame='galactic')
    sfdmap = np.empty(hp.pixelfunc.nside2npix(2048))
    sfdmap[pix2048] = sfd(coords)
    return sfdmap

def get_gnilc_map():
    pix2048 = np.arange(hp.pixelfunc.nside2npix(2048))
    selpixang = hp.pixelfunc.pix2ang(2048, pix2048, lonlat=True)
    coords = SkyCoord(l=selpixang[0] * units.deg, b=selpixang[1] * units.deg, frame='galactic')
    gnilcmap = np.empty(hp.pixelfunc.nside2npix(2048))
    gnilcmap[pix2048] = gnilc(coords)
    return gnilcmap

'''
def get_accdict_with_references(resultdictnames, runtitles, n_bootstrap=1):
    reconnewlist = []
    for rfname in resultdictnames:
        reconnewlist.append(pickle.load(open(rfname, 'rb'))) #'UnitTests/gnupartmp2/test0.pkl'
    
    #sfdmap
    print('ckp-1')
    sfdmap = get_sfd_map() #some mem inten stuff here / slow, presave the sfd query too?
    print('ckp0')
    #b17map
    b17map = get_bayestar2017_map()
    print('ckp1')
    gc.collect()
    dustmaplist = [sfdmap, b17map]
    for resultdict in reconnewlist:
        dustmaplist.append(resultdict['dustmap'])
        assert np.allclose(resultdict['reconpix'], reconnewlist[0]['reconpix']) #check that all have the same pixmasks
    dustmaplist = np.vstack(dustmaplist)
    print('ckp2')
    accdict = get_acc_nside2048_batch(dustmaplist, reconnewlist[0]['reconpix'], n_bootstrap=n_bootstrap, device="cpu", dtype=torch.float64)
    namelist = ['SFD', 'Bayestar17']+runtitles
    accdict.update({'mapnames': namelist})
    accdict.update({'n_bootstrap':n_bootstrap})
    accdict.update({'resultdictnames': resultdictnames})
    return accdict
'''

def get_sfd_error(reconmap, sfdmap, pix):
    residual = reconmap[pix] - sfdmap[pix]
    mean, std = np.mean(residual), np.std(residual)
    print('Mean = {:.3f} Std = {:.3f}'.format(mean, std))
    return mean, std

def get_sfd_errorplot(reconmap, sfdmap, pix, args):
    residual = reconmap[pix] - sfdmap[pix]
    mean, std = np.mean(residual), np.std(residual)
    print('Mean = {:.3f} Std = {:.3f}'.format(mean, std))
    fig, ax = plt.subplots()
    ax.scatter(sfdmap[pix], reconmap[pix], s=1)
    ax.plot(np.linspace(np.min(sfdmap[pix]), np.max(sfdmap[pix]), 50), np.linspace(np.min(sfdmap[pix]), np.max(sfdmap[pix]), 50), linestyle='dashed', c='k')
    plt.xlabel('SFD')
    plt.ylabel('Map Input')
    if 'xlim' in args.keys():
        ax.set_xlim(args['xlim'])
    if 'ylim' in args.keys():
        ax.set_ylim(args['ylim'])
    plt.show()
    return mean, std


def plot_with_sfd_presaved_2048(dustmap, selpix, rot=[0, 90], suptitle=None, xsize=3000):
    '''
    dustmap: Nside=2048 healpix array with recon vals at selpix indices
    selpix are the pixel ids at 2048
    '''

    sfdmap = np.zeros(hp.pixelfunc.nside2npix(2048))
    with h5py.File('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/reference_maps/bayestar2017.hdf5', 'r') as f:
        b17mean = np.array(f['mean'])
        coords_lb = np.array(f['coords_lb'])

    pix2048 = hp.pixelfunc.ang2pix(2048, coords_lb[:, 0], coords_lb[:, 1], lonlat=True)
    b17map = np.empty(hp.pixelfunc.nside2npix(2048))
    b17map[pix2048] = b17mean
    del b17mean
    '''
    with h5py.File('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/reference_maps/bayestar2019.hdf5', 'r') as f:
        b19mean = np.array(f['mean'])
        coords_lb = np.array(f['coords_lb'])
    
    pix2048 = hp.pixelfunc.ang2pix(2048, coords_lb[:, 0], coords_lb[:, 1], lonlat=True)
    b19map = np.empty(hp.pixelfunc.nside2npix(2048))
    b19map[pix2048] = b19mean
    del b19mean
    '''
    # sfd
    selpixang = hp.pixelfunc.pix2ang(2048, selpix, lonlat=True)
    coords = SkyCoord(l=selpixang[0] * units.deg, b=selpixang[1] * units.deg, frame='galactic')
    sfdmap[selpix] = sfd(coords)
    print('Recon offset rel SFD')
    get_sfd_error(dustmap, sfdmap, selpix)
    print('B17 offset rel SFD')
    get_sfd_error(b17map, sfdmap, selpix)
    
    #print('SFD, SNR = {:.3f}'.format(np.mean(sfdmap[selpix])/np.std(sfdmap[selpix])))
    #print('B17, SNR = {:.3f}'.format(np.mean(b17map[selpix])/np.std(b17map[selpix])))
    #print('Recon, SNR = {:.3f}'.format(np.mean(dustmap[selpix])/np.std(dustmap[selpix])))
    
    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    ax = ax.ravel()
    plt.axes(ax[0])
    hp.gnomview(dustmap, rot=rot, hold=True, title='Recon', xsize=xsize, min=np.min(sfdmap), max=np.max(sfdmap))
    plt.axes(ax[1])
    hp.gnomview(sfdmap, rot=rot, hold=True, title='SFD', xsize=xsize, min=np.min(sfdmap), max=np.max(sfdmap))
    plt.axes(ax[2])
    hp.gnomview(b17map, rot=rot, hold=True, title='Bayestar17', xsize=xsize, min=np.min(sfdmap), max=np.max(sfdmap))
    plt.delaxes(ax[3])
    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.show()
    return sfdmap, b17map#, b19map



###Testbed Functions###########
def get_pixint_histograms(reglist, fullmaps):
    for reg in reglist:
        plt.figure()
        for (mname, fullmap) in fullmaps:
            plt.hist(fullmap[reg['coords']], label=mname, alpha=0.2, density=True)
        plt.legend()
        plt.suptitle(reg['name'])
        plt.show()
    return

def get_masked_patch(fullmap, reconpix, nres=2048):
    mmap = np.ones(hp.nside2npix(nres))*hp.UNSEEN
    mmap[reconpix] = fullmap[reconpix]
    return mmap

def get_testbed_dict(name, Nresol=2048):
    if name=='Perseus':
        pl, pb = np.array([157.5, 161.5, 161.5, 157.5]), np.array([-22, -22, -16, -16])
        pverts = hp.ang2vec(pl, pb, lonlat=True)
        pix2k = hp.query_polygon(Nresol, pverts, inclusive=True)
        rot, xsize = [159.5, -19], 300
    elif name=='Ursa Major':
        pl, pb = np.array([140, 162, 162, 140]), np.array([32, 32, 44, 44])
        pverts = hp.ang2vec(pl, pb, lonlat=True)
        pix2k = hp.query_polygon(Nresol, pverts, inclusive=True)
        rot, xsize = [151, 38], 800
    elif name=='Polaris':
        pl, pb = np.array([119, 128, 128, 119]), np.array([22, 22, 34, 34])
        pverts = hp.ang2vec(pl, pb, lonlat=True)
        pix2k = hp.query_polygon(Nresol, pverts, inclusive=True)
        rot, xsize= [124, 28], 800
    elif name=='Pegasus':
        pl, pb = np.array([85, 100, 100, 85]), np.array([-42, -42, -28, -28])
        pverts = hp.ang2vec(pl, pb, lonlat=True)
        pix2k = hp.query_polygon(Nresol, pverts, inclusive=True)
        rot, xsize= [92.5, -36], 800
    elif name=='Stripe-Splotch':
        tstripe4 = hp.ang2pix(4, 135, -30, lonlat=True)
        tstripe32 = get_smallpix_in_tilepix(4, tstripe4, 32)
        pix2k = []
        for t32 in tstripe32:
            pix2k.append(get_smallpix_in_tilepix(32, t32, Nresol))
        pix2k = np.unique(np.hstack(pix2k))
        rot, xsize= [135, -30], 1000
    elif name=='NGC':
        pix2k = get_tile_idx_in_circlepatch(Nresol, [0, 90], 50)
        rot, xsize = [0, 90], 5000    
    elif name=='NGC_bgt35':
        pix2k = get_tile_idx_in_circlepatch(Nresol, [0, 90], 55)
        rot, xsize = [0, 90], 5000
        
    else:
        raise NotImplementedError
    return {'name':name, 'coords': pix2k, 'Nresol': Nresol, 'rot': rot, 'xsize': xsize}


def plot_maps_comparison(testbedlist, compmaps, norm='SFD'):
    #cols: maps, rows: regions
    nrows, ncols = len(testbedlist), len(compmaps)
    
    if norm=='SFD':
        if compmaps[0][0]=='SFD':
            sfdmap = compmaps[0][1]
        else:
            sfdmap = get_sfd_map()
    
    
    fig, ax = plt.subplots(figsize=(20, 20), nrows=nrows, ncols=ncols)
    for (rix, reg) in enumerate(testbedlist):
        if norm=='SFD':
            vmin, vmax = np.min(sfdmap[reg['coords']]), np.max(sfdmap[reg['coords']])
            pargs = {'min': vmin, 'max': vmax}
        else:
            pargs = {'norm': 'hist'}
            
        for (cix, mtup) in enumerate(compmaps):
            plt.axes(ax[rix, cix])
            hp.gnomview(mtup[1], rot=testbedlist[rix]['rot'], hold=True, title=reg['name']+': '+mtup[0], xsize=testbedlist[rix]['xsize'], **pargs)
    plt.suptitle('Comparison')
    plt.show()
    return

def plot_acc_comparison(accsref, accobjlist, cols, ref_choice=[0, 1, 2, 3], ylim=[-0.5, 0.5]):
    plt.figure(figsize=(10,4))
    for m in range(len(accsref)):
        if m in ref_choice:
            errens = accsref[m][-1]
            sampstd = 1000*np.sqrt(np.mean(errens**2, axis=0))
            res = 1000*accsref[m][1]['accs'].flatten()
            lb, ub = 0 - sampstd, sampstd
            plt.fill_between(accsref[m][1]['mean_z']+1, -(2*sampstd), (2*sampstd), alpha=0.1, color=cols[m])
            plt.fill_between(accsref[m][1]['mean_z']+1, lb, ub, alpha=0.2, color=cols[m])
            plt.plot(accsref[m][1]['mean_z']+1, res, label=accsref[m][0], color=cols[m])
    ctr=m+1
    if accobjlist is not None:
        for accnew in accobjlist:
            for m in range(len(accnew)):#m=0, in most cases
                errens = accnew[m][-1]
                sampstd = 1000*np.sqrt(np.mean(errens**2, axis=0))
                res = 1000*accnew[m][1]['accs'].flatten()
                lb, ub = 0 - sampstd, sampstd
                plt.fill_between(accnew[m][1]['mean_z']+1, -(2*sampstd), (2*sampstd), alpha=0.1, color=cols[ctr])
                plt.fill_between(accnew[m][1]['mean_z']+1, lb, ub, alpha=0.2, color=cols[ctr])
                plt.plot(accnew[m][1]['mean_z']+1, res, label=accnew[m][0], color=cols[ctr])
                ctr +=1

    plt.xscale("log")
    plt.xticks(np.arange(5)+1,np.arange(5))
    plt.ylabel(r"$\delta E_{B-V}\:[mmag]$")
    plt.xlabel(r"$z$")
    plt.hlines(0,*plt.gca().get_xlim(),color="k", linestyle='dashed')
    plt.legend()
    plt.ylim(ylim)
    plt.show()
    return

def visualize_cuts_multirow(df_input, cutfunc, cut_args, attributes, get_mask=False, figsize=(8, 8)):
    '''
    df_input: cuts that have been made upto that point
    cutfunc: function from astro_cuts
    cut_args: arguments for cutfunc
    attributes: list of [('hist', cutfield), ('scatter', ['field_x', ['field_y']])] with any plot related attr in third element of tuple
    get_mask: return mask
    '''
    fig, ax = plt.subplots(nrows=len(attributes), ncols=2, figsize=figsize)
    #rows: field, columns: before vs after cut
    if cut_args is not None:
        mask = cutfunc(df_input, **cut_args)
    else:
        mask = cutfunc(df_input)
    df_after = df_input.iloc[mask, :]
    for r, rtuple in enumerate(attributes):
        if rtuple[0]=='hist':
            cutfield = rtuple[1]
            if len(rtuple)>2:
                ax[r][0].hist(df_input[cutfield].to_numpy(), density=True, bins=20, alpha=0.4, label='All', **rtuple[2])
                ax[r][0].hist(df_input.iloc[~mask, :][cutfield].to_numpy(), density=True, bins=20, alpha=0.4, label='Removed', **rtuple[2])
                ax[r][1].hist(df_after[cutfield].to_numpy(), alpha=0.4, density=True, bins=20, **rtuple[2])
            else:
                ax[r][0].hist(df_input[cutfield].to_numpy(), density=True, bins=20, alpha=0.4, label='All')
                ax[r][0].hist(df_input.iloc[~mask, :][cutfield].to_numpy(), density=True, bins=20, alpha=0.4, label='Removed')
                ax[r][1].hist(df_after[cutfield].to_numpy(), alpha=0.4, density=True, bins=20)
            ax[r][0].legend()
            ax[r][0].set_xlabel(cutfield)
            ax[r][1].set_xlabel(cutfield)
            ax[r][0].set_title('Before cuts')
            ax[r][1].set_title('After cuts')
            
        if rtuple[0]=='scatter':
            field_x, field_y = rtuple[1]
            ax[r][0].scatter(df_input[field_x].to_numpy(), df_input[field_y].to_numpy(), s=1, label='All')
            ax[r][0].scatter(df_input.iloc[~mask, :][field_x].to_numpy(), df_input.iloc[~mask, :][field_y].to_numpy(), s=1, label='Removed')
            ax[r][1].scatter(df_after[field_x].to_numpy(), df_after[field_y].to_numpy(), s=1)
            ax[r][0].legend()
            ax[r][0].set_xlabel(field_x)
            ax[r][1].set_xlabel(field_x)
            ax[r][0].set_ylabel(field_y)
            ax[r][1].set_ylabel(field_y)
            if len(rtuple)>2:
                if 'xlim' in rtuple[2].keys():
                    ax[r][0].set_xlim(rtuple[2]['xlim'])
                    ax[r][1].set_xlim(rtuple[2]['xlim'])
                if 'ylim' in rtuple[2].keys():
                    ax[r][0].set_ylim(rtuple[2]['ylim'])
                    ax[r][1].set_ylim(rtuple[2]['ylim'])
                if 'invert_y' in rtuple[2].keys():
                    ax[r][0].invert_yaxis()
                    ax[r][1].invert_yaxis()
    print('Remaining fraction = {:.3f}'.format(np.sum(mask)/len(mask)))

    print('Check bias on E_median / E_sigma--------------------------')
    print('Median E_median before cuts', np.median(df_input['E_median'].to_numpy()))
    print('Median E_median after cuts', np.median(df_after['E_median'].to_numpy()))
    print('Median E_sigma before cuts', np.median(df_input['E_sigma'].to_numpy()))
    print('Median E_sigma after cuts', np.median(df_after['E_sigma'].to_numpy()))
    plt.show()
    if get_mask:
        return df_after, mask
    else:
        return df_after

