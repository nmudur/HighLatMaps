import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py

from astropy.coordinates import SkyCoord
import astropy.units as units

from sys import platform
from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery
from dustmaps.planck import PlanckGNILCQuery

if platform=='linux':
    gnilc = PlanckGNILCQuery()
    sys.path.append('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code_Nresol/')
else:
    sys.path.append('../methods_code_Nresol/')


from utils_circpatch import *  # check if import works
from do_recon_tilewise import *

sfd = SFDQuery()


def get_bayestar2017_map(return_sigma=False, conversion_factor=0.856):
    if platform=='linux':
        bstarpath = '/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/reference_maps/'
    else:
        bstarpath = '../../../reference_maps/'

    if return_sigma:
        with h5py.File(bstarpath+'bayestar2017_wsigma.hdf5', 'r') as f:
            b17mean = np.array(f['mean'])
            b17sigma = np.array(f['sigma'])
            coords_lb = np.array(f['coords_lb'])
    else:
        with h5py.File(bstarpath+'bayestar2017.hdf5', 'r') as f:
            b17mean = np.array(f['mean'])
            coords_lb = np.array(f['coords_lb'])

    pix2048 = hp.pixelfunc.ang2pix(2048, coords_lb[:, 0], coords_lb[:, 1], lonlat=True)
    b17map = np.empty(hp.pixelfunc.nside2npix(2048))
    b17map[pix2048] = b17mean
    print(f'Multiplying by the EBV conversion factor of {conversion_factor} derived from E(g-r) relation (Eq1)')
    if return_sigma:
        b17sigma_map = np.empty(hp.pixelfunc.nside2npix(2048))
        b17sigma_map[pix2048] = b17sigma
        return conversion_factor*b17map, conversion_factor*b17sigma_map
    else:
        return conversion_factor*b17map

    
    
def get_bayestar2019_map(conversion_factor=0.856):
    print(platform)
    if platform=='linux':
        bstarpath = '/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/reference_maps/'
    else:
        bstarpath = '../../../reference_maps/'
    
    with h5py.File(bstarpath+'bayestar2019.hdf5', 'r') as f:
        b19mean = np.array(f['mean'])
        coords_lb = np.array(f['coords_lb'])

    pix2048 = hp.pixelfunc.ang2pix(2048, coords_lb[:, 0], coords_lb[:, 1], lonlat=True)
    b19map = np.empty(hp.pixelfunc.nside2npix(2048))
    print(f'Multiplying by the EBV conversion factor of {conversion_factor} derived from E(g-r) relation (Eq1)')

    b19map[pix2048] = b19mean
    return conversion_factor*b19map


def get_sfd_map():
    pix2048 = np.arange(hp.pixelfunc.nside2npix(2048))
    selpixang = hp.pixelfunc.pix2ang(2048, pix2048, lonlat=True)
    coords = SkyCoord(l=selpixang[0] * units.deg, b=selpixang[1] * units.deg, frame='galactic')
    sfdmap = np.empty(hp.pixelfunc.nside2npix(2048))
    sfdmap[pix2048] = 0.86*sfd(coords)
    print('Multiplying by the blue tip correction factor of 0.86')
    return sfdmap #value at index p is the value of SFD queried at pixel p at Nside=2048

def get_gnilc_map(conversion_factor=1.0):
    pix2048 = np.arange(hp.pixelfunc.nside2npix(2048))
    selpixang = hp.pixelfunc.pix2ang(2048, pix2048, lonlat=True)
    coords = SkyCoord(l=selpixang[0] * units.deg, b=selpixang[1] * units.deg, frame='galactic')
    gnilcmap = np.empty(hp.pixelfunc.nside2npix(2048))
    gnilcmap[pix2048] = gnilc(coords)
    print(f'Multiplying by the EBV conversion factor of {conversion_factor}, the slope of SFD*0.86 with GNILC at pixhigh')
    return conversion_factor*gnilcmap

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



def plot_maps_comparison(testbedlist, compmaps, norm='SFD', figsize=(20, 20), kwargs_dict=None):
    #cols: maps, rows: regions
    nrows, ncols = len(testbedlist), len(compmaps)
    
    if norm=='SFD':
        if compmaps[0][0]=='SFD':
            sfdmap = compmaps[0][1]
        else:
            sfdmap = get_sfd_map()
    
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    for (rix, reg) in enumerate(testbedlist):
        if norm=='SFD':
            vmin, vmax = np.min(sfdmap[reg['coords']]), np.max(sfdmap[reg['coords']])
            pargs = {'min': vmin, 'max': vmax}
        elif norm=='minmax':
            assert 'norm_percs' in kwargs_dict.keys()
            minperc, maxperc = kwargs_dict['norm_percs']
            irefmap = kwargs_dict['peg_to_map_idx']
            
            refmap = compmaps[irefmap][1]
            arr = hp.gnomview(refmap, rot=reg['rot'], xsize =reg['xsize'], return_projected_map=True, no_plot=True)
            assert np.sum(arr==hp.UNSEEN)==0
            vmin, vmax = np.percentile(arr, minperc), np.percentile(arr, maxperc)
            print('Color pegged to Map, Min, Max', compmaps[irefmap][0], vmin, vmax)
            pargs = {'min': vmin, 'max': vmax}
        else:
            pargs = {'norm': 'hist'}
            
        for (cix, mtup) in enumerate(compmaps):
            if (nrows==1):
                plt.axes(ax[cix])
            else:
                plt.axes(ax[rix, cix])
            resinfo = False if cix==0 else True
            hp.gnomview(mtup[1], rot=testbedlist[rix]['rot'], hold=True, title=reg['name']+': '+mtup[0], xsize=testbedlist[rix]['xsize'], notext=resinfo, **pargs)
    plt.tight_layout()
    if 'savefig' in kwargs_dict.keys():
        fig.savefig(kwargs_dict['savefig'], dpi=200)
    plt.show()
    return

def query_at_lbcoords(lbpoint, compmaps, Nsideresol):
    pixel = hp.ang2pix(Nsideresol, *lbpoint, lonlat=True)
    for m, mtup in enumerate(compmaps):
        assert len(mtup[1])==hp.nside2npix(Nsideresol)
        print(mtup[0], ': ', mtup[1][pixel])
    return
              

def plot_acc_comparison(accsref, accobjlist, cols, ref_choice=[0, 1, 2, 3], ylim=[-0.5, 0.5], sigcontours='Default', no_legend=False, savefig=None, title=None):
    fig = plt.figure(figsize=(10,5))
    '''if legendnames=='Default':
        maplabels = [accsref[m][0] for m in range(len(accsref))]
        for accnew in accobjlist:
            maplabels += [accnew[m][0] for m in range(len(accnew))] #handling the case where the input has multiple maps' accs
    else:
        assert #number of maps to be plotted must match the number of '''
    for m in range(len(accsref)):
        if m in ref_choice:
            errens = accsref[m][-1]
            sampstd = 1000*np.sqrt(np.mean(errens**2, axis=0)) #RMSE: variability about 0
            res = 1000*accsref[m][1]['accs'].flatten()
            lb, ub = 0 - sampstd, sampstd
            if sigcontours=='Default': #1, 2 rms deviations
                plt.fill_between(accsref[m][1]['mean_z']+1, -(2*sampstd), (2*sampstd), alpha=0.1, color=cols[m])
                plt.fill_between(accsref[m][1]['mean_z']+1, lb, ub, alpha=0.2, color=cols[m])
            else:
                assert sorted(sigcontours)==sigcontours #asc order eg: [1, 3, 5]
                maxalpha=0.3
                for icont, sigfac in enumerate(sigcontours):
                    plt.fill_between(accsref[m][1]['mean_z']+1, -(sigfac*sampstd), (sigfac*sampstd), alpha=maxalpha-(icont*0.1), color=cols[m])
            plt.plot(accsref[m][1]['mean_z']+1, res, label=accsref[m][0], color=cols[m], marker='.')
            if no_legend:
                print(accsref[m][0])
    ctr=m+1
    if accobjlist is not None:
        for accnew in accobjlist:
            for m in range(len(accnew)):#m=0, in most cases
                errens = accnew[m][-1]
                sampstd = 1000*np.sqrt(np.mean(errens**2, axis=0))
                res = 1000*accnew[m][1]['accs'].flatten()
                lb, ub = 0 - sampstd, sampstd
                if sigcontours=='Default': #1, 2 rms deviations
                    plt.fill_between(accnew[m][1]['mean_z']+1, -(2*sampstd), (2*sampstd), alpha=0.1, color=cols[ctr])
                    plt.fill_between(accnew[m][1]['mean_z']+1, lb, ub, alpha=0.2, color=cols[ctr])
                else:
                    assert sorted(sigcontours)==sigcontours #asc order eg: [1, 3, 5]
                    maxalpha=0.3
                    for icont, sigfac in enumerate(sigcontours):
                        plt.fill_between(accnew[m][1]['mean_z']+1, -(sigfac*sampstd), (sigfac*sampstd), alpha=maxalpha-(icont*0.1), color=cols[ctr])
                plt.plot(accnew[m][1]['mean_z']+1, res, label=accnew[m][0], color=cols[ctr], marker='.')
                if no_legend:
                    print(accnew[m][0])
                ctr +=1

    plt.xscale("log")
    plt.xticks(np.arange(5)+1,np.arange(5))
    plt.ylabel(r"$\delta E_{B-V}\:[mmag]$")
    plt.xlabel(r"$z$")
    plt.hlines(0,*plt.gca().get_xlim(),color="k", linestyle='dashed')
    if not no_legend:
        plt.legend()
    plt.ylim(ylim)
    if title is not None:
        fig.suptitle(title, y=0.94)
    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
    return


def plot_acc_comparison_sampling(accobjlist, cols, ylim=[-0.5, 0.5], sigcontours='Default', no_legend=False, savefig=None, title=None):
    '''
    :param accobjlist: List of accdict['accs_all']
    :param cols: Colors
    :param ylim:
    :param no_legend:
    :param savefig:
    :param title:
    :return:
    '''

    fig = plt.figure(figsize=(10,5))
    ctr=0
    for accnew in accobjlist: #each element is the "accs_all" element of the output of preproc_get_acc_intgn
        for m in range(len(accnew)): #m=0, in most cases (When multiple maps' accs were passed to preproc_get_acc_intgn at the same time, it's >1)
            #res = 1000*accnew[m][1]['accs'].flatten() #not this because this has shape 1xNbtsxNz
            stdbts = 1000*accnew[m][-1].flatten()
            meanbts = 1000 * accnew[m][-2].flatten()

            if sigcontours=='Default': #1, 2 sig deviations about the mean
                plt.fill_between(accnew[m][1]['mean_z']+1, meanbts-(2*stdbts), meanbts+(2*stdbts), alpha=0.1, color=cols[ctr])
                plt.fill_between(accnew[m][1]['mean_z']+1, meanbts-stdbts, meanbts+stdbts, alpha=0.2, color=cols[ctr])
            else:
                assert sorted(sigcontours)==sigcontours #asc order eg: [1, 3, 5]
                maxalpha=0.3
                for icont, sigfac in enumerate(sigcontours):
                    plt.fill_between(accnew[m][1]['mean_z']+1, meanbts-(sigfac*stdbts), meanbts+(sigfac*stdbts), alpha=maxalpha-(icont*0.1), color=cols[ctr])
            plt.plot(accnew[m][1]['mean_z']+1, meanbts, label=accnew[m][0], color=cols[ctr], marker='.')
            if no_legend:
                print(accnew[m][0])
            ctr +=1

    plt.xscale("log")
    plt.xticks(np.arange(5)+1,np.arange(5))
    plt.ylabel(r"$\delta E_{B-V}\:[mmag]$")
    plt.xlabel(r"$z$")
    plt.hlines(0,*plt.gca().get_xlim(),color="k", linestyle='dashed')
    if not no_legend:
        plt.legend()
    plt.ylim(ylim)
    if title is not None:
        fig.suptitle(title, y=0.94)
    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
    return


def plot_errorbar_ratio(accobjdict, cols, no_legend=False, savefig=None, title=None):
    '''
    :param accobjdict: 'Rotated', 'Bootstrapped'

        e.g accsfddict = pickle.load(open('runs/2_6/accdir/sfd_corr_acc_intgn.pkl', 'rb'))
            accsfd = accsfddict['accs_all']
            accbtsdict = pickle.load(open('runs/2_6/accdir/panel_acc_intgn_bootstrapped.pkl', 'rb'))
            accbts = accbtsdict['accs_all'] #has length number of maps whose accs were evaluated by that routine
            #for only SFD
            {'Rotated': [accsfd[0]], 'Bootstrapped': [accbts[0]]}
            accsfd[0] & accbts[0] have length 5

    :param cols: Colors
    :param ylim:
    :param no_legend:
    :param savefig:
    :param title:
    :return:
    '''
    fig = plt.figure(figsize=(10,5))

    #Rotated EBs
    accrotlist = accobjdict['Rotated']
    accbtslist = accobjdict['Bootstrapped']
    assert len(accrotlist)== len(accbtslist)

    for m in range(len(accrotlist)):
        accrot = accrotlist[m]
        accbts = accbtslist[m]
        print('Rotated Map: ', accrot[0])
        #assert len(accrot) == 5
        errens = accrot[-1]
        rmsrot = 1000*np.sqrt(np.mean(errens**2, axis=0))

        #Bootstrapped EBs
        print('Bootstrapped Map: ', accbts[0])
        assert len(accbts) == 5
        stdbts = 1000*accbts[-1].flatten()

        plt.plot(accrot[1]['mean_z']+1, rmsrot/stdbts, marker='.', color=cols[m], label=accrot[0])


    plt.xscale("log")
    plt.xticks(np.arange(5)+1,np.arange(5))
    plt.ylabel(r"$\delta E_{B-V}\:[mmag]$")
    plt.xlabel(r"$z$")

    if not no_legend:
        plt.legend(prop={'size': 10})
    if title is not None:
        fig.suptitle(title, y=0.94)
    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
    return

def view_map_patch(Nside, selpix, rot, fullmap, xsize=500, view='gnomview', title=None):
    selmap = np.ones(hp.nside2npix(Nside))*hp.UNSEEN
    selmap[selpix] = fullmap[selpix]
    
    if view=='gnomview':
        hp.gnomview(selmap, rot=rot, xsize=xsize, title=title)
    else:
        hp.mollview(selmap, rot=rot, title=title)
        
    return


def get_sfd_error(reconmap, sfdmap, pix, printout=True):
    if np.isnan(reconmap[pix]).sum()!=0:
        print('Nan at {} pixels'.format(np.isnan(reconmap[pix]).sum()))
        pix = pix[~np.isnan(reconmap[pix])]
        
    residual = reconmap[pix] - sfdmap[pix]
    mean, std = np.mean(residual), np.std(residual)
    if printout:
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


def get_binned_meanstar_properties(testbed, Nsideresol, fieldlist, STARFILE= '/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/data/lsdraw/stars_edr3/{}.fits'):
    '''
    returns maps with mean posterior properties binned to Nsideresol
    testbed: Region (output of get_testbed_dict)
    Nsideresol: Pixelization to which starproperties are to be binned
    fieldlist: which columns to take the mean of -- either a column in starfile or 'z' (z-score)
    STARFILE: starfile format
    returns: maskpix: pixels which contained stars
             maps: list of maps with the mean of the stars' properties binned to Nsideresol
    '''
    maps= []
    maskpix = []
    for im in range(len(fieldlist)):
        maps.append(np.ones(hp.nside2npix(Nsideresol))*hp.UNSEEN)
    #get sfd for z score
    if 'z' in fieldlist:
        assert Nsideresol==2048 #because you're querying sfd at the same nsideresol as the starmap is being binned to
        sfdmap = get_sfd_map()
    #which starfiles need to be queried
    t16 = np.unique(get_largepix_for_smallpix(testbed['coords'], testbed['Nresol'], 16))
    print('# Nside=16 pixels', len(t16))
    #for each starfile
    for tile in t16:
        pix2k = get_smallpix_in_tilepix(16, tile, Nsideresol)
        dfstars = get_stars_within_tile(16, tile, convert_to_dataframe(STARFILE.format(tile)))
        dfstars['pix2048'] = hp.ang2pix(Nsideresol, dfstars['l'].to_numpy(), dfstars['b'].to_numpy(), lonlat=True)
        for im, field in enumerate(fieldlist):
            if field=='z':
                dfstars['z'] = (dfstars['E_median'].to_numpy() - sfdmap[dfstars['pix2048'].to_numpy()]) / dfstars['E_sigma'].to_numpy()
            dfpwise = dfstars[['pix2048', field]].groupby('pix2048').mean()
            maps[im][dfpwise.index.values] = dfpwise[field].to_numpy()
        maskpix.append(dfpwise.index.values)
        if tile%100==0:
            print('{} Pix16 done'.format(tile))
    return np.hstack(maskpix), maps

#Testbeds.py related plots

def plot_noise_vs_latitude(latwise_offsets, kwargs):
    '''
    :param latwise_offsets: List of outputs of MapComparisons.get_sfd_offset_noise_for_patches
        latiwseoffsets[ilat]: output for latitude[il] and is a list of length num_compmaps. Output of get_testbeds_latitudewise
        latiwseoffsets[ilat][imap]: offset data for latitude[il] and map[im]
    :return:
    '''
    plot_map_names = [mapwelem[0] for mapwelem in latwise_offsets[0]]
    figsize=kwargs['figsize'] if 'figsize' in kwargs.keys() else (6, 6)
    
    plt.figure(figsize=figsize)
    cycle = ['b', 'k', 'r', 'g', 'y'] #plt.rcParams['axes.prop_cycle'].by_key()['color']
    alpha=1
    for il, latlist in enumerate(latwise_offsets):
        assert len(latlist) == len(plot_map_names)
        for m, mapwiseoffsets in enumerate(latlist):
            offset_dict = mapwiseoffsets[1]
            lat_value = int(offset_dict['set_name'][offset_dict['set_name'].rindex('~')+1:])
            offset_std = offset_dict['Offset_std']
            if 'errorbars' in kwargs.keys():
                alpha=0.2
                plt.errorbar(lat_value, np.mean(offset_std), np.std(offset_std), c=cycle[m])
                plt.scatter(lat_value, np.mean(offset_std), marker='d', label=(plot_map_names[m] if il==0 else None), c=cycle[m])
                
            if il==0:
                plt.scatter(np.ones(len(offset_std))*lat_value, offset_std, s=1, label=(plot_map_names[m] if 'errorbars' not in kwargs.keys() else None), c=cycle[m], alpha=alpha)
            else:
                #assert mapnames are in the same order
                plt.scatter(np.ones(len(offset_std)) * lat_value, offset_std, s=1, c=cycle[m], alpha=alpha)
    
    legnd = plt.legend()
    legnd.legendHandles[0]._sizes = [10]
    legnd.legendHandles[1]._sizes = [10]
    legnd.legendHandles[2]._sizes = [10]
    
    plt.xlabel('Latitude')
    plt.ylabel(r'$\sigma$(Map - SFD)')
    plt.title('Std of Map- SFD in Nside=32 pixels at different latitudes')
    if 'ylim' in kwargs.keys():
        plt.ylim(kwargs['ylim'])
    if 'savefig' in kwargs.keys():
        plt.savefig(**kwargs['savefig'])
    
    plt.show()
    return


def plot_noise_vs_stellardensity(starwise_offsets, kwargs):
    '''
    :param plot_map_names: Names of the maps for the plot
    :param starwise_offsets: List of outputs of MapComparisons.get_sfd_offset_noise_for_patches
        starwiseoffsets[ilat]: output for stdensity[il] and is a list of length num_compmaps. Output of get_testbeds_latitudewise
        starwiseoffsets[ilat][imap]: offset data for stdensity[il] and map[im]
    :return:
    '''
    figsize=kwargs['figsize'] if 'figsize' in kwargs.keys() else (6, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    cycle = ['b', 'k'] #plt.rcParams['axes.prop_cycle'].by_key()['color']
    binvals = []
    for il, binlist in enumerate(starwise_offsets):
        assert len(binlist) == len(plot_map_names)
        for m, mapwiseoffsets in enumerate(binlist):
            offset_dict = mapwiseoffsets[1]
            bin_value = int(float(offset_dict['set_name'][offset_dict['set_name'].rindex('=')+1:]))
            if m==0:
                binvals.append(bin_value) 
            offset_std = offset_dict['Offset_std']
            if il==0:
                plt.scatter(np.ones(len(offset_std))*bin_value, offset_std, label=plot_map_names[m], s=1, c=cycle[m])
            else:
                plt.scatter(np.ones(len(offset_std)) * bin_value, offset_std, s=1, c=cycle[m])
    plt.legend()
    plt.xlabel('Stars per Nside=32 pixel')
    plt.ylabel(r'$\sigma$(Map - SFD)')
    plt.xscale('log')
    xticklabels = kwargs['labels'] if 'labels' in kwargs.keys() else binvals
    print(xticklabels)
    ax.set_xticks(binvals, xticklabels, rotation=90)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.title('Std of Map- SFD in Nside=32 pixels in a given region')
    if 'savefig' in kwargs.keys():
        plt.savefig(kwargs['savefig'])
    plt.show()
    return

def plot_z_scores_vs_region(mapwise_zscores, kwargs):
    '''
    :param mapwise_zscores: Output of get_zscores_for_patches
    :return:
    '''
    plot_map_names = [zsc[0] for zsc in mapwise_zscores]
    figsize=kwargs['figsize'] if 'figsize' in kwargs.keys() else (6, 6)
    numbins = kwargs['bins'] if 'bins' in kwargs.keys() else 20
    dpi = 100 if 'dpi' not in kwargs.keys() else kwargs['dpi']
    #what bins?
    if 'peg_bins' in kwargs.keys():
        data = mapwise_zscores[0][1]['z-scores'] if kwargs['peg_bins'] == 'First' else np.hstack([mapz[1]['z-scores'] for mapz in mapwise_zscores])
        bins = np.histogram(data, bins=numbins)[1]
    else:
        bins=numbins

    plt.figure(figsize=figsize, dpi = dpi)
    region_name = mapwise_zscores[0][1]['set_name']
    for m, mapwise_zscore_tup in enumerate(mapwise_zscores):
        mapwise_zscore = mapwise_zscore_tup[1]
        assert mapwise_zscore['combined'] #assuming not a list of patches
        assert mapwise_zscore['set_name'] == region_name #making sure the same region? better way to do this?

        plt.hist(mapwise_zscore['z-scores'], bins=bins, label=plot_map_names[m], alpha=0.5, density=True, zorder=kwargs['zorder'][m] if 'zorder' in kwargs.keys() else None)

        if 'mean/std' in kwargs.keys():
            print('{}: Mean={:.3f}, Std={:.3f}'.format(plot_map_names[m], np.mean(mapwise_zscore['z-scores']), np.std(mapwise_zscore['z-scores'])))
    plt.xlabel(r'$\frac{Map - SFD}{\sigma(Map)}$')
    title = kwargs['title'] if 'title' in kwargs.keys() else '\'Z-Score\' distribution for pixels in {}'.format(region_name) 
    plt.title(title)
    if 'savefig' in kwargs.keys():
        plt.savefig(kwargs['savefig'])
    plt.legend()
    plt.show()
    return


def plot_noise_vs_region(combined_offset_noise_for_patches, kwargs):
    '''
    :param mapwise_offsets: Output of MapComparisons.get_sfd_offset_noise_for_patches_combined for ONE input region
    :return:
    '''

    region_name = combined_offset_noise_for_patches['set_name']
    mapreslist = combined_offset_noise_for_patches['mapwise_offset']
    plot_map_names = [mtup[0] for mtup in mapreslist]
    mapwise_offsets = [mtup[1] for mtup in mapreslist] #list of arrays for each map. Each array element corresponds to the offset for the map in that pixel

    figsize=kwargs['figsize'] if 'figsize' in kwargs.keys() else (6, 6)
    dpi = 100 if 'dpi' not in kwargs.keys() else kwargs['dpi']
    numbins = kwargs['bins'] if 'bins' in kwargs.keys() else 20

    #what bins to use
    if 'peg_bins' in kwargs.keys():
        data = mapwise_offsets[0] if kwargs['peg_bins'] == 'First' else np.hstack([mapoff for mapoff in mapwise_offsets])
        bins = np.histogram(data, bins=numbins)[1]
    else:
        bins = numbins
    
    plt.figure(figsize=figsize, dpi=dpi)
    for m, mapwise_offset in enumerate(mapwise_offsets):
        plt.hist(mapwise_offset, bins=bins, label=plot_map_names[m], alpha=0.5, density=True, zorder=kwargs['zorder'][m] if 'zorder' in kwargs.keys() else None)
        if 'mean/std' in kwargs.keys():
            print('{}: Mean={:.3f}, Std={:.3f}'.format(plot_map_names[m], np.mean(mapwise_offset), np.std(mapwise_offset)))
    plt.xlabel('Map - SFD')
    title = kwargs['title'] if 'title' in kwargs.keys() else 'Offset distribution for pixels in {}'.format(region_name)
    plt.title(title)
    if 'savefig' in kwargs.keys():
        plt.savefig(kwargs['savefig'])
    plt.legend()
    plt.show()
    return