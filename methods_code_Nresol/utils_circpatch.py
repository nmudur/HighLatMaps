import healpy
import numpy as np
import healpy as hp
import pickle
import h5py
import os
#from astropy-healpix import HEALPix

'''
Funcs for creating and checking (l, b) circles around hp nside centers
'''

def check_all_verts_in_circle_pix(nsidetile, tile, radius_fac):
    #check that all vertices of the given pixel at a given nsidetile are inside the circumcircle w radius resolution*radius_fac
    pixcenlb = hp.pixelfunc.pix2ang(nsidetile, tile, lonlat=True)
    radius_rad = hp.pixelfunc.nside2resol(nsidetile)*radius_fac
    radius_degsq = (radius_rad * 180/np.pi)**2
    vertices = hp.boundaries(nside=nsidetile, pix=tile, step=1)
    cornlb = hp.pixelfunc.vec2ang(vertices.T, lonlat=True)
    distcen = (cornlb[0] - pixcenlb[0])**2 + (cornlb[1] - pixcenlb[1])**2
    checkbool = np.all(distcen<radius_degsq)
    return checkbool, distcen, radius_degsq

def check_all_verts_in_circle_region(nsidetile, radius_fac, pixlist):
    #=range(hp.pixelfunc.nside2npix(nsidetile)) if all pixels on sphere
    #check that all vertices of the given pixel at a given nsidetile are inside the circumcircle
    for tile in pixlist:
        boolout, distcen, radius_degsq = check_all_verts_in_circle_pix(nsidetile, tile, radius_fac)
        if ~boolout:
            print(tile, distcen, radius_degsq)
        return

def get_tile_idx_in_lbpatch(nsidetile, boxlb):
    #return all the Nside=nsidetile pixels corresponding to the bounding box boxlb
    lmin, bmin, lmax, bmax = boxlb
    regv = []
    regv.append(hp.pixelfunc.ang2vec(lmin, bmin, lonlat=True))
    regv.append(hp.pixelfunc.ang2vec(lmin, bmax, lonlat=True))
    regv.append(hp.pixelfunc.ang2vec(lmax, bmax, lonlat=True))
    regv.append(hp.pixelfunc.ang2vec(lmax, bmin, lonlat=True))
    selpix = hp.query_polygon(nsidetile, regv, inclusive=True)
    return selpix

# Compare overlap ratio
def compare_overlap(nsidetile, radius_fac, boxlb, Nsideresol=2048):
    seltiles = get_tile_idx_in_lbpatch(nsidetile, boxlb)

    numpix = np.zeros(len(seltiles))
    for t, tile in enumerate(seltiles):
        pixcenvec = hp.pixelfunc.pix2vec(nsidetile, tile)
        radius_rad = hp.pixelfunc.nside2resol(nsidetile ) *radius_fac
        disccoords = hp.query_disc(Nsideresol, pixcenvec, radius_rad)
        numpix[t] = len(disccoords)
    print("Num Pixels with tiling", np.sum(numpix))

    lmin, bmin, lmax, bmax = boxlb
    regv = []
    regv.append(hp.pixelfunc.ang2vec(lmin, bmin, lonlat=True))
    regv.append(hp.pixelfunc.ang2vec(lmin, bmax, lonlat=True))
    regv.append(hp.pixelfunc.ang2vec(lmax, bmax, lonlat=True))
    regv.append(hp.pixelfunc.ang2vec(lmax, bmin, lonlat=True))
    selpix = hp.query_polygon(Nsideresol, regv)
    print('Without tiling', len(selpix))
    print('Excess ratio from double counting', np.sum(numpix ) /len(selpix))
    return

def get_lbbeam_idx_for_tile(nsidetile, tile, radius_fac, Nsideresol=2048, dirname=None, protocol=pickle.HIGHEST_PROTOCOL):
    #saves / returns region matrix in vectors for the circle centered around pixel = tile at Nsidetile
    pixcenvec = hp.pixelfunc.pix2vec(nsidetile, tile)
    radius_rad = hp.pixelfunc.nside2resol(nsidetile) * radius_fac
    pixindisc = hp.query_disc(Nsideresol, pixcenvec, radius_rad)
    region = hp.pixelfunc.pix2vec(Nsideresol, pixindisc)
    region = np.vstack(region).T
    if dirname is not None:
        np.save(dirname+'region_'+str(tile), region)
        pickle.dump({'nsidetile': nsidetile, 'tile': tile, 'cen_lb': hp.pixelfunc.pix2ang(nsidetile, tile, lonlat=True),
                     'radius_region': np.rad2deg(radius_rad), 'radius_stars': np.rad2deg(radius_rad)+25/60}, open(dirname+'patchdata_'+str(tile)+'.pkl', 'wb'), protocol=protocol)
    return region #replace with h5py/fits and figure out how to put info in m
'''
#OLD pixelvec which misses or overcounts some of the pixels
def get_pixelvec_for_tile(nsidetile, tile, dirname=None, protocol=pickle.HIGHEST_PROTOCOL):
    #saves / returns region matrix in vectors for pixel = tile at Nsidetile
    #radius_deg in the pickle is irrelevant if you're note saving the pickle

    pixcenvec = np.array(hp.pixelfunc.pix2vec(nsidetile, tile))
    corners = healpy.boundaries(nsidetile, tile, step=1).T
    regpix = hp.query_polygon(2048, corners, inclusive=True) #maybe do query disc and then remove extras?
    region = hp.pixelfunc.pix2vec(2048, regpix)
    region = np.vstack(region).T
    radius_deg = np.rad2deg(np.max(np.arccos(np.matmul(pixcenvec, corners.T)))) + 25/60
    #radius_rad = #np.max( of np.abs( of pixcenvec - corners plus the 25/60 for 5 arcmin thing

    if dirname is not None:
        np.save(dirname+'region_'+str(tile), region)
        pickle.dump({'nsidetile': nsidetile, 'tile': tile, 'cen_lb': hp.pixelfunc.pix2ang(nsidetile, tile, lonlat=True),
                     'radius_stars_deg': radius_deg}, open(dirname+'patchdata_'+str(tile)+'.pkl', 'wb'), protocol=protocol)
    return region #replace with h5py/fits and figure out how to put info in m


'''

def get_largepix_for_smallpix(pixhigh, highresnside, lowresnside):
    '''
    :param pixhigh: pix_2048 array. Ring ordered.
    :param highresnside: 2048
    :param lowresnside: Nsidetile
    highres, lowres = 32, 16
    pixhigh = hp.nest2ring(32, np.arange(4))
    hp.ring2nest(lowres, get_largepix_for_smallpix(pixhigh, highres, lowres)) # 0 0 0 0
    :return:
    '''
    #pix32, 32, 16
    nsfac = int((highresnside / lowresnside)**2)  # 4, 32->16
    nestedhighpix = hp.ring2nest(highresnside, pixhigh)  # 0
    nestedlowpix = nestedhighpix // nsfac
    #print('Nested', nestedlowpix)
    return hp.nest2ring(lowresnside, nestedlowpix)

def get_smallpix_in_tilepix(nsidetile, tile, res):
    nsfac = int((res / nsidetile)**2)  # 2, 16->32
    nestedtile = hp.ring2nest(nsidetile, tile)  # 0
    smallpixnested = np.arange(nsfac * nestedtile, nsfac * (nestedtile + 1), dtype=int)  # 0, 1, 2, 3
    return hp.nest2ring(res, smallpixnested)

def get_pixelvec_for_tile(nsidetile, tile, Nsideresol, dirname=None, protocol=pickle.HIGHEST_PROTOCOL):
    #saves / returns region matrix in vectors for pixel = tile at Nsidetile
    #radius_deg in the pickle is irrelevant if you're not saving the pickle
    smallpixid = get_smallpix_in_tilepix(nsidetile, tile, Nsideresol)
    region = np.vstack(hp.pixelfunc.pix2vec(Nsideresol, smallpixid)).T

    if dirname is not None:
        #fix for saving later
        pixcenvec = np.array(hp.pixelfunc.pix2vec(nsidetile, tile))
        corners = healpy.boundaries(nsidetile, tile, step=1).T
        radius_deg = np.rad2deg(np.max(np.arccos(np.matmul(pixcenvec, corners.T)))) + 25 / 60
        np.save(dirname+'region_'+str(tile), region)
        pickle.dump({'nsidetile': nsidetile, 'tile': tile, 'cen_lb': hp.pixelfunc.pix2ang(nsidetile, tile, lonlat=True),
                     'radius_stars_deg': radius_deg}, open(dirname+'patchdata_'+str(tile)+'.pkl', 'wb'), protocol=protocol)

    return region #replace with h5py/fits and figure out how to put info in m




def get_tile_idx_in_circlepatch(nsidetile, lbcen, radius_deg):
    #return all the Nside=nsidetile pixels corresponding to the bounding box boxlb
    radius_rad = np.deg2rad(radius_deg)
    cen_vec = hp.pixelfunc.ang2vec(lbcen[0], lbcen[1], lonlat=True)
    tiles = hp.query_disc(nsidetile, cen_vec, radius_rad, inclusive=True)
    return tiles


def get_2048_healpix_map_from_gnuparpatches(dirname):
    #repackaging hdf5 files into a single 2048 healpix array
    #sort of unsafe because you're assuming no mixups in the pixel order that generated the hdf5 files (which should ideally not have happened since you're using get_smallpix... everywhere)
    know=0
    if os.path.exists(dirname+'recon_stats'):
        recon_info = pickle.load(open(dirname+'recon_stats', 'rb'))
        Nsidetile = recon_info['Nsidetile']
        if 'tiles' in recon_info.keys():
            print('from pickle')
            tiles = recon_info['tiles']
            know=1
    if know==0:
        print('from hdf5 files')
        reconfiles = [f for f in os.listdir(dirname) if f.endswith('.hdf5')]
        tiles = [int(st[st.index('recon_')+6: st.index('.')]) for st in reconfiles]
    print('Ckp')
    selpix = []
    reconmap = np.zeros(hp.pixelfunc.nside2npix(2048))
    varmap = np.zeros(hp.pixelfunc.nside2npix(2048))
    for tile in tiles:
        smallpix = get_smallpix_in_tilepix(Nsidetile, tile, 2048)
        selpix.append(smallpix)
        smallpixang = hp.pixelfunc.pix2ang(2048, smallpix, lonlat=True)
        with h5py.File(dirname+'recon_{}.hdf5'.format(tile), 'r') as f:
            reconmap[smallpix] = np.array(f['recon'])
            varmap[smallpix] = np.array(f['variance'])
    selpix = np.hstack(selpix)
    return reconmap, varmap, selpix


def get_Nsideresol_healpix_map_from_gnuparpatches_safe(dirname, Nsideresol=2048, tiles=None):
    #repackaging hdf5 files into a single 2048 healpix array
    if tiles is None:
        recon_info = pickle.load(open(dirname+'recon_info.pkl', 'rb'))
        if 'tiles' in recon_info.keys():
            tiles = recon_info['tiles']
        else:
            reconfiles = [f for f in os.listdir(dirname) if f.endswith('.hdf5')]
            tiles = [int(st[st.index('recon_')+6: st.index('.')]) for st in reconfiles]

    print('Ckp')
    selpix = []
    reconmap = np.zeros(hp.pixelfunc.nside2npix(Nsideresol))
    varmap = np.zeros(hp.pixelfunc.nside2npix(Nsideresol))
    corr = []
    for tile in tiles:
        try:
            with h5py.File(dirname+'recon_{}.hdf5'.format(tile), 'r') as f:
                coordvec = np.array(f['vec'])
                smallpix = hp.pixelfunc.vec2pix(Nsideresol, coordvec[:, 0], coordvec[:, 1], coordvec[:, 2])
                reconmap[smallpix] = np.array(f['recon'])
                varmap[smallpix] = np.array(f['variance'])
                selpix.append(smallpix)
        except OSError:
            corr.append(tile)
    if len(corr)>0:
        print('Missing / Corrupted files: ', corr)
    else:
        selpix = np.hstack(selpix)
        return reconmap, varmap, selpix



def get_stars_within_tile(Nsidetile, tile, df):
    #retrieve only stars lying within a pixel
    starpix = hp.ang2pix(Nsidetile, df['l'].to_numpy(), df['b'].to_numpy(), lonlat=True)
    mask = starpix==tile
    return df.iloc[mask, :]    
    
    
    
if __name__=='__main__':
    '''
    #Perseus patchtest
    Nsidetile = 16
    tile = hp.pixelfunc.ang2pix(Nsidetile, 159, -21, lonlat= True)
    check, distcen, rad = check_all_verts_in_circle_pix(Nsidetile, tile, 0.8)
    lmin, bmin, lmax, bmax = 0, 30, 18, 60
    compare_overlap(Nsidetile, 0.8, [lmin, bmin, lmax, bmax])

    #going with radius_fac = 0.8
    region = get_lbbeam_idx_for_tile(Nsidetile, tile, 0.8, dirname='Data/PerseusSingleTileCheck/')
    '''
    #Tile test
    '''
    Nsidetile = 16
    tile = hp.pixelfunc.ang2pix(Nsidetile, 159, -21, lonlat=True)
    region = get_pixelvec_for_tile(Nsidetile, tile, Nsideresol=2048, dirname = 'Data/Experiment2B/perseus_tile_')
    print(3)

    '''


    #1/20th test
    '''
    Nsidetile = 16
    tiles = get_tile_idx_in_lbpatch(Nsidetile, [0, 30, 18, 60])
    print(len(tiles), tiles)

    for tile in tiles:
        get_pixelvec_for_tile(Nsidetile, tile, Nsideresol=2048, dirname='Data/Experiment2B/', protocol=2)
    


    Nsidetile = 16
    tiles = get_tile_idx_in_lbpatch(Nsidetile, [0, 30, 18, 60])
    print(len(tiles), tiles)

    #zerothid = get_respix_in_tile(Nsidetile, hp.pixelfunc.nest2ring(Nsidetile, 0), 32)

    for tile in tiles:
        region = get_pixelvec_for_tile(Nsidetile, tile, Nsideresol=2048)
        #smallpixid = get_respix_in_tile(Nsidetile, tile, 32)
    '''
    highres, lowres = 32, 16
    pixhigh = hp.nest2ring(32, np.arange(4))
    print(hp.ring2nest(lowres, get_largepix_for_smallpix(pixhigh, highres, lowres)))

    # Checking looptilewise
    highres, lowres = 2048, 32
    coords2048 = np.array([0, 1, 2, 3])  # 0th pix at Nside32
    print(hp.ring2nest(lowres, get_largepix_for_smallpix(coords2048, highres, lowres)))
