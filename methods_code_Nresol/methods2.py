import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import jv
from random import randint
import subprocess

from utils_circpatch import *
from do_recon_tilewise import *
from methods_cluster import *

##Eval-level (probably put in another module): basically looptilewise / gnupartilewise which are called as func(coords)
#equivalent of nearest


code_dirname = '/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code_Nresol/'


##Full region-level functions: Inputs: coords, recon_func_name, cuts_list, kwargs_dict: recon_kwargs, outer_kwargs (for region level functions)
#Call the tile-level functions or work on tiling internally. Stars are only queried here if tiling is done here.
    
def gnupartilewise_new(coords, recon_func_name, cuts_list, kwargs_dict, runname, sleep=180, mem=2000, compile_pickle=True):
    '''
    coords: Pixels at Nsideresol (default=2048)
    recon_func_name: methods_cluster function to use for the reconstruction
    cuts_list: List of astro_cuts functions and their arguments
    kwargs_dict: Dictionary containing:
            outer_kwargs: Nsideresol: Resolution of the map
                          stars_presaved (optional): which set of presaved LSD files. Presaved LSD tiles are at Nside=16
                          Nsidetile: Sub-region level pixel resolution
                          radius_deg_extra: Padding around star_tiles (see do_recon_tilewise.cuts_wrapper)
    runname: Result name and output tmpdir
    '''
    # Same as above but for when the reconstruction is likely to require staggering
    tmpdir = runname + '_tmpdir/'
    if os.path.exists(tmpdir):
        print('Dir already exists')
    else:
        os.mkdir(tmpdir)
    
    if not os.path.exists(os.path.join(tmpdir, 'logs')):
        os.mkdir(os.path.join(tmpdir, 'logs'))

    # tiles: Nsidetile pixels corresponding to coords
    tiles = get_largepix_for_smallpix(coords, kwargs_dict['outer_kwargs']['Nsideresol'], kwargs_dict['outer_kwargs']['Nsidetile'])
    tiles = np.unique(tiles)
    jhash = randint(100, 999)
    
    #Generate list of tiles
    if os.path.exists(tmpdir + 'tiles.txt'): #only relevant if this is a staggered run (ran the same command twice with diff mem requests)
        print('Tiles file already exists')
        missing = []
        for tile in tiles:
            if not os.path.exists(tmpdir+'recon_{}.hdf5'.format(tile)):
                missing.append(tile)
        print('Only reconstructing for missing {} tiles'.format(len(missing)))
        tilefile = tmpdir + 'tiles_jhash{}.txt'.format(jhash) 
        with open(tilefile, 'w') as ft:
            for tile in missing:
                ft.write(str(tile) + '\n')



    else:
        tilefile = tmpdir + 'tiles.txt'
        with open(tilefile, 'w') as ft:
            for tile in tiles:
                ft.write(str(tile) + '\n')

    #save pickle with run specifications that are read in by do_recon_tilewise
    recon_info = {'Nsidetile': kwargs_dict['outer_kwargs']['Nsidetile'],
                  'radius_deg_extra': kwargs_dict['outer_kwargs']['radius_deg_extra'],
                  'Nsideresol': kwargs_dict['outer_kwargs']['Nsideresol'],
                  'recon_kwargs': kwargs_dict['recon_kwargs'],
                  'cuts_list': cuts_list,
                  'tiles': tiles,
                  'recon_func_name': recon_func_name}
    #Added later: if you don't want to use the default STARDIR. Eg: For outer_gaia check.
    if 'stars_presaved' in kwargs_dict['outer_kwargs'].keys():
        if isinstance(kwargs_dict['outer_kwargs']['stars_presaved'], str):
            print('Using dir ', kwargs_dict['outer_kwargs']['stars_presaved'])
        
        stars_presaved = kwargs_dict['outer_kwargs']['stars_presaved']
        recon_info.update({'stars_presaved': kwargs_dict['outer_kwargs']['stars_presaved']})

    if os.path.exists(tmpdir + 'recon_info.pkl'):
        print('Tmp pickle already exists')
    else:
        pickle.dump(recon_info, open(tmpdir + 'recon_info.pkl', 'wb'))



    # subprocess to .sh
    parstr = 'sbatch --error=' + tmpdir + str(jhash) + '.e --output=' + tmpdir + str(
        jhash) + '.o --export=ALL,TMPDIR=' + tmpdir + ',TILEFILE=' + tilefile + ',MEM=' + str(
        mem) + ' ' + code_dirname + 'parallel.sbatch'
    out = subprocess.run('module load parallel && ' + parstr, check=True, capture_output=True, shell=True)

    # MAIN
    if compile_pickle:
        time.sleep(sleep)
        reconmap, varmap, reconpix = get_Nsideresol_healpix_map_from_gnuparpatches_safe(tmpdir, Nsideresol=kwargs_dict['outer_kwargs']['Nsideresol'], tiles=tiles)
        result_dict = {'dustmap': reconmap, 'variancemap': varmap, 'reconpix': reconpix,
                       'func': 'gnupartilewise_new',
                       'cuts_list': cuts_list, 'recon_func_name': recon_func_name,
                       'kwargs_dict': kwargs_dict}
        pickle.dump(result_dict, open(runname + '.pkl', 'wb'))
        return result_dict
    else:
        return
    
    
    
    
#Loop_tilewise
def looptilewise(coords, recon_func_name, cuts_list, kwargs_dict, runname, save_tilewise=False, compile_pickle=True):
    '''
    :param coords: assuming healpix pixel ids at Nsideresol
    :param recon_func_name:
    :param cuts_list: eg:
    :param kwargs_dict:
        'recon_kwargs' and 'outer_kwargs'
    :return:
    '''
    #for now this is gonna reconstruct more than the coords that were asked for. fix later if needed
    tmpdir = runname+'_tmpdir/'
    Nsideresol = kwargs_dict['outer_kwargs']['Nsideresol']
    Nsidetile, radius_deg_extra = kwargs_dict['outer_kwargs']['Nsidetile'], kwargs_dict['outer_kwargs']['radius_deg_extra']
    tiles = get_largepix_for_smallpix(coords, Nsideresol, Nsidetile)
    tiles = np.unique(tiles)
    print('Looping over {} tiles'.format(len(tiles)))
    reconmap, varmap = np.ones(hp.pixelfunc.nside2npix(Nsideresol))*hp.UNSEEN, np.ones(hp.pixelfunc.nside2npix(Nsideresol))*hp.UNSEEN
    reconpix = []
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    
    if 'stars_presaved' in kwargs_dict['outer_kwargs'].keys():
        if isinstance(kwargs_dict['outer_kwargs']['stars_presaved'], str):
            print('Using dir ', kwargs_dict['outer_kwargs']['stars_presaved'])
        stars_presaved = kwargs_dict['outer_kwargs']['stars_presaved']
    else:
        stars_presaved = '/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/data/lsdraw/stars/{}.fits'
        
    for tile in tiles:
        recon, variance, coordvec = get_recon_for_tile(tile, Nsidetile, radius_deg_extra, getattr(methods_cluster, recon_func_name), kwargs_dict['recon_kwargs'], cuts_list, Nsideresol, save=save_tilewise, savdirname=tmpdir, presaved= stars_presaved, return_vec=True) #edited
        smallpix = hp.pixelfunc.vec2pix(Nsideresol, coordvec[:, 0], coordvec[:, 1], coordvec[:, 2])
        reconmap[smallpix] = recon
        varmap[smallpix] = variance
        reconpix.append(smallpix)
    
    reconpix = np.hstack(reconpix)
    if compile_pickle:
        result_dict = {'dustmap': reconmap, 'variancemap': varmap, 'reconpix': reconpix,
                       'func': 'looptilewise',
                       'cuts_list': cuts_list, 'recon_func_name': recon_func_name,
                       'kwargs_dict': kwargs_dict} #TODO: Check what else goes here. Do Unit test.
        pickle.dump(result_dict, open(runname+'.pkl', 'wb'))
        return result_dict
    else:
        return