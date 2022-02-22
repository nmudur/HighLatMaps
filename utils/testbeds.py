import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import h5py

from astropy.coordinates import SkyCoord
import astropy.units as units

from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery
from dustmaps.planck import PlanckGNILCQuery

from sys import platform
if platform=='linux':
    sys.path.append('/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code_Nresol/')
else:
    sys.path.append('../methods_code_Nresol/')

from utils_circpatch import *
from do_recon_tilewise import *
from vis_utils import *

def get_pixels_in_Bayestar_footprint(Nside):
    #returns only pixels at Nside that are completely contained in the Bayestar footprint
    tiles_all = np.arange(hp.nside2npix(Nside))
    b17map = get_bayestar2017_map()
    nanmask = np.isnan(b17map)
    covered_tiles= []
    for tile in tiles_all:
        pix_in_tile = get_smallpix_in_tilepix(Nside, tile, 2048)
        if np.sum(nanmask[pix_in_tile])==0:
            covered_tiles.append(tile)
    return np.array(covered_tiles)

def get_subset_pixels_at_latitude(Nsidetile, tile_set, latitude, Numoutput):
    '''
    :param Nsidetile:
    :param tile_set: set of pixels at Nsidetile. eg: output of get_pixels_in_Bayestar_footprint
    :param latitude: input latitude
    #for absolute just run twice and concatenate
    :param Numoutput: Number of pixels to return at that latitude
    :return:
    '''
    lats = hp.pix2ang(Nsidetile, tile_set, lonlat=True)[1]
    diff = np.abs(lats - latitude)
    closestmask = (diff == np.min(diff)) #pixels from the set that are nearest to the input latitude
    assert np.sum(closestmask)>=Numoutput #possible pixels are more than the number of pixels you wanna return
    closest_pixels = tile_set[closestmask]
    #sorted longitudes
    closest_pixels_sorted = closest_pixels[np.argsort(hp.pix2ang(Nsidetile, closest_pixels, lonlat=True)[0])]
    output_subset = closest_pixels_sorted[np.linspace(0, len(closest_pixels_sorted)-1, Numoutput+1).astype('int')][:-1]
    #dont include the 360'th degree longitude since that's next to 0 -- so query one extra
    return output_subset


def get_testbeds_latitudewise(latitude, footprint, Numoutput=4, Nresol=2048):
    '''
    :param latitude:
    :param footprint: footprint function
    :param Nresol: Resolution
    :return:
    '''
    assert Nresol==2048 #for now since all the footprints are defined in terms of 2048
    NSIDETILE = 32
    footprint_tiles = get_pixels_in_Bayestar_footprint(NSIDETILE)
    tiles = get_subset_pixels_at_latitude(NSIDETILE, footprint_tiles, latitude, Numoutput)
    output = []
    for tile in tiles:
        output.append({'tile': tile, 'coords': get_smallpix_in_tilepix(NSIDETILE, tile, 2048), 'lonlat': hp.pix2ang(NSIDETILE, tile, lonlat=True), 'Nsideresol': Nresol})
    output_dict = {'footprint': footprint.__name__, 'set_name': 'Latitude~{}'.format(latitude), 'Nsidetile': NSIDETILE, 'Nsideresol': Nresol}
    output_dict.update({'patches': output})
    return output_dict


class MapComparisons():
    def __init__(self, compmaps, Nsideresol=2048):
        '''
        :param compmaps: List of (mapname, map)
        :param Nsideresol: Map resolution
        '''
        self.maps = compmaps
        #reference maps
        self.sfdmap = get_sfd_map()
        self.b17map = get_bayestar2017_map(return_sigma=True) #tuple with mean, sigma

        assert Nsideresol==2048

        self.with_sigma = True if len(self.maps[0])==3 else False #flag on whether the loaded maps have sigmas

        for im, mtup in enumerate(self.maps):
            assert len(mtup[1])==hp.nside2npix(Nsideresol)
            assert len(mtup)==3 if self.with_sigma else 2 #ensures all maps either do or dont have uncertainties

    def get_sfd_offset_noise_for_patches(self, testbed_set):
        '''
        :param testbed_set: Output of eg: get_testbeds_latitudewise
        :return: List of length number of maps
        '''
        mapwise_sfd_offsets = [] #list of length len(self.maps)

        for mtuple in self.maps:
            if self.with_sigma:
                mname, meanmap, sigmamap = mtuple
            else:
                mname, meanmap = mtuple
            print(mname)
            resdict = {'set_name': testbed_set['set_name']} #separate result for each map
            print(testbed_set['set_name'])
            offset_for_each_patch = []
            patches = testbed_set['patches'] #list
            offset_mean, offset_std = [], []
            for patch in patches: #patch = Nside=32 pixel in the testbed
                mean, std = get_sfd_error(meanmap, self.sfdmap, patch['coords'], printout=True)
                offset_mean.append(mean)
                offset_std.append(std)
            resdict.update({'Offset_mean': np.array(offset_mean), 'Offset_std': np.array(offset_std)})
            mapwise_sfd_offsets.append((mname, resdict))
        return mapwise_sfd_offsets


    def get_z_scores_for_patches(self, testbed_set, combined=False):
        '''
        :param testbed_set: Output of eg: get_testbeds_latitudewise
        :returns: the zscore of the patch relative to SFD = (recon_mean - sfd)/sigma
        '''
        assert self.with_sigma
        mapwise_zscores = []

        for mname, meanmap, sigmamap in self.maps:
            print(mname)
            resdict = {'set_name': testbed_set['set_name']} #separate result for each map
            zscores_patches = []
            for patch in testbed_set['patches']:
                z_score = (meanmap[patch['coords']] - self.sfdmap[patch['coords']])/sigmamap[patch['coords']]
                zscores_patches.append(z_score)
            if combined:
                zscores_patches = np.hstack(zscores_patches)
            resdict.update({'z-scores': zscores_patches, 'combined': combined})
            mapwise_zscores.append((mname, resdict))
        return mapwise_zscores




if __name__=='__main__':
    '''
    #testing querying a band of pixels within a predefined subset at a given latitude
    NSIDETILE = 32
    bstpix32 = get_pixels_in_Bayestar_footprint(NSIDETILE)
    pix32 = get_subset_pixels_at_latitude(NSIDETILE, tile_set=bstpix32, latitude=20, Numoutput=12)
    print(hp.pix2ang(NSIDETILE, pix32, lonlat=True))
    print(pix32)
    '''

    #Plotting offset_noise_for different latitude ranges: (multiple outputs of get_testbeds_latitudewise)
    '''
    b17map = get_bayestar2017_map()
    recondict = pickle.load(open('../reconmaps/16a_fwhm-6-1__bgt21.pkl', 'rb'))
    recon16a = recondict['dustmap']
    compmaps = [('B17', b17map), ('16a', recon16a)]

    mapcomp = MapComparisons(compmaps)

    latrange = np.arange(30, 100, 10).astype('int')
    latrange = np.insert(latrange, 0, 25)
    latwise_offsets = []
    for lat in latrange:
        Numtiles = 4 if lat==90 else 10
        patches = get_testbeds_latitudewise(lat, get_pixels_in_Bayestar_footprint, Numoutput=Numtiles, Nresol=2048)
        latwise_offsets.append(mapcomp.get_sfd_offset_noise_for_patches(patches))
    plot_noise_vs_latitude(['Bayestar2017', '16a'], latwise_offsets, {})
    '''

    #Plotting zscores for different maps for a SINGLE latitude
    b17map, b17sigmamap = get_bayestar2017_map(return_sigma=True)
    recondict = pickle.load(open('../reconmaps/16a_fwhm-6-1__bgt21.pkl', 'rb'))
    recon16a = recondict['dustmap']
    sigmamap = np.sqrt(recondict['variancemap']) #check if issues here

    compmaps = [('B17', b17map, b17sigmamap), ('16a', recon16a, sigmamap)]

    mapcomp = MapComparisons(compmaps)
    latrange = np.arange(30, 100, 10).astype('int')
    latrange = np.insert(latrange, 0, 25)
    latwise_offsets = []
    for lat in latrange:
        patches = get_testbeds_latitudewise(lat, get_pixels_in_Bayestar_footprint, Numoutput=10, Nresol=2048)
        zscores = mapcomp.get_z_scores_for_patches(patches, combined=True)
        plot_z_scores_vs_region(['B17', '16a'], zscores, {})
    print(32)

