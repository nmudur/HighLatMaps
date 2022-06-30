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

def get_pixels_in_recon61xbayestar_footprint(Nside):
    b17map = get_bayestar2017_map()
    nbmaindir = '/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/notebooks_main1/'
    footprint = pickle.load(open(nbmaindir+'fullsky_runs/footprints/ftp_minimal_3_12.pkl', 'rb'))
    #selects only pixels at Nside that are completely contained in the Recon61 footprint
    fptpixels = footprint['pixels']
    fptpixels = fptpixels[~np.isnan(b17map[fptpixels])]
    #selects only pixels at Nside that are ALSO completely contained in the B17 footprint
    if Nside==2048:
        return fptpixels
    else:
        tiles_all = np.arange(hp.nside2npix(Nside))
        exclusionmask = ~np.isin(np.arange(hp.nside2npix(2048)), fptpixels)
        covered_tiles= []
        for tile in tiles_all:
            pix_in_tile = get_smallpix_in_tilepix(Nside, tile, 2048)
            if np.sum(exclusionmask[pix_in_tile])==0:
                covered_tiles.append(tile)
    return np.array(covered_tiles)


def get_subset_pixels_at_latitude(Nsidetile, tile_set, latitude, Numoutput='max'):
    '''
    :param Nsidetile:
    :param tile_set: set of 'allowed' pixels at Nsidetile. eg: output of get_pixels_in_Bayestar_footprint
    :param latitude: input latitude
    #for absolute just run twice and concatenate
    :param Numoutput: Number of pixels to return at that latitude
    :return:
    '''
    lats = hp.pix2ang(Nsidetile, tile_set, lonlat=True)[1]
    diff = np.abs(lats - latitude)
    #pixels from the set that are nearest to the input latitude
    closestmask = (diff == np.min(diff))
    closest_pixels = tile_set[closestmask]
    #sorted longitude-wise
    closest_pixels_sorted = closest_pixels[np.argsort(hp.pix2ang(Nsidetile, closest_pixels, lonlat=True)[0])]
    if Numoutput=='max':
        output_subset = closest_pixels_sorted
    else:
        assert isinstance(Numoutput, int)
        assert np.sum(closestmask)>=Numoutput #possible pixels are more than the number of pixels you wanna return
        #pick Numoutput pixels with equal distribution in longitude
        output_subset = closest_pixels_sorted[np.linspace(0, len(closest_pixels_sorted)-1, Numoutput+1).astype('int')][1:]
    #dont include the 360'th degree longitude since that's next to 0 -- so query one extra and leave the last one out
    return output_subset


def get_testbeds_latitudewise(latitude, footprint, Numoutput, Nresol=2048):
    '''
    :param latitude:
    :param footprint: footprint function
    :param Nresol: Resolution
    :return:
    '''
    assert Nresol==2048 #for now since all the footprints are defined in terms of 2048
    NSIDETILE = 32
    footprint_tiles = footprint(NSIDETILE)
    tiles = get_subset_pixels_at_latitude(NSIDETILE, footprint_tiles, latitude, Numoutput)
    assert len(tiles)==np.unique(len(tiles)) #make sure no duplicate patches
    output = []
    for tile in tiles:
        output.append({'tile': tile, 'coords': get_smallpix_in_tilepix(NSIDETILE, tile, 2048), 'lonlat': hp.pix2ang(NSIDETILE, tile, lonlat=True), 'Nsideresol': Nresol})
    output_dict = {'footprint': footprint.__name__, 'set_name': 'Latitude~{}'.format(latitude), 'Nsidetile': NSIDETILE, 'Nsideresol': Nresol}
    output_dict.update({'patches': output})
    return output_dict

def get_testbeds_stellardensitywise(stdensity_bins, Nsidetile, tiles, tilewise_binid, tilesperbin='max', Nresol=2048):
    '''
    :param stdensity_bins: Array with the lower, upper bounds of the bins (number of stars)
    :param Nsidetile: Nside of the pixels in tiles
    :param tiles: Nside=32 pixels
    :param tilewise_binid: Bin ID assignment corresponding to stdensity_bins for each tile in tiles
    :param tilesperbin: Number of Nside=32 pixel to assign to each stellar density bin
    :param Nresol:
    :return: List of length (stdensity bins) of dicts of the same format as get_testbeds_latitudewise
    eg:
    bin_id = np.digitize(numstars_postcuts, bins)
    binlow = np.insert(bins, np.floor(np.min(numstars_postcuts)/100)*100, 0)
    binupp = np.append(bins, np.ceil(np.max(numstars_postcuts)/100)*100)
    stdensity_bins : [binlow, binupp]
    tilewise_binid : bin_id
    '''
    assert len(tiles) == len(tilewise_binid)
    unibins, counts = np.unique(tilewise_binid, return_counts=True)
    assert np.min(counts)>tilesperbin

    binlow, binup = stdensity_bins[0], stdensity_bins[1]
    binmid = (binup - binlow)/2 + binlow
    assert len(unibins) == len(binlow) == len(binup) #as many bin ids as there are bins

    testbedlist = []

    for ist, stbinmid in enumerate(binmid):
        stdict = {'set_name': 'StBinMid={}'.format(stbinmid), 'Nsidetile': Nsidetile, 'Nsideresol': Nresol, 'Bin_bnds': (binlow[ist], binup[ist])}
        stmask = (tilewise_binid==ist) #which tiles have this bin id
        tiles_in_bin = tiles[stmask]
        
        if isinstance(tilesperbin, int):
            rng = np.random.default_rng(seed=100)
            maskout = rng.choice(len(tiles_in_bin), tilesperbin,replace=False)
            tiles_in_bin = tiles_in_bin[maskout]
            assert len(tiles_in_bin) == len(np.unique(tiles_in_bin))
        #else take all
        output = []
        for tile in tiles_in_bin:
            output.append({'tile': tile, 'coords': get_smallpix_in_tilepix(Nsidetile, tile, 2048), 'lonlat': hp.pix2ang(Nsidetile, tile, lonlat=True), 'Nsideresol': Nresol})
        stdict.update({'patches': output})
        testbedlist.append(stdict)
    return testbedlist, stdensity_bins


def check_assignment(numstars_postcuts, bin_id, stdensitybins):
    binlow, binup = stdensitybins
    assert len(numstars_postcuts) == len(bin_id)
    assert len(binlow) == len(binup) == len(np.unique(bin_id))

    for ib, stbin in enumerate(binlow):
        relmask = (bin_id==ib)
        assert np.all(numstars_postcuts[relmask]>=stbin), stbin
        assert np.all(numstars_postcuts[relmask]<binup[ib]), stbin
    return


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
    elif name=='Stripe-Square':
        pix2k = get_tile_idx_in_lbpatch(Nresol, [246, 26, 274, 54])
        rot, xsize = [260, 40], 1000
    elif name=='NGC':
        pix2k = get_tile_idx_in_circlepatch(Nresol, [0, 90], 50)
        rot, xsize = [0, 90], 5000
    elif name=='NGC_bgt90':
        pix2k = get_tile_idx_in_circlepatch(Nresol, [0, 90], 10)
        rot, xsize = [0, 90], 5000
    elif name=='NGC_bgt35':
        pix2k = get_tile_idx_in_circlepatch(Nresol, [0, 90], 55)
        rot, xsize = [0, 90], 5000
    elif name=='FullSky_Bayestar':
        assert Nresol==2048
        b17map = get_bayestar2017_map()
        notnan = ~np.isnan(b17map)
        coords = np.arange(hp.nside2npix(2048))
        assert len(coords) == len(b17map)
        pix2k = coords[notnan]
        rot= [0, 90]
        xsize= 5000
    elif name=='FullSky_Bayestar_babsgt20':
        assert Nresol==2048
        b17map = get_bayestar2017_map()
        notnan = ~np.isnan(b17map)
        coords = np.arange(hp.nside2npix(2048))
        lbang = hp.pix2ang(2048, coords, lonlat=True)
        latmask = (np.abs(lbang[1])>20)
        assert len(coords) == len(b17map)
        notnan *= latmask
        pix2k = coords[notnan]
        
        rot= [0, 90]
        xsize= 5000
    elif name=='FullSky_Bayestar_babsgt19':
        assert Nresol==2048
        b17map = get_bayestar2017_map()
        notnan = ~np.isnan(b17map)
        coords = np.arange(hp.nside2npix(2048))
        lbang = hp.pix2ang(2048, coords, lonlat=True)
        latmask = (np.abs(lbang[1])>19)
        assert len(coords) == len(b17map)
        notnan *= latmask
        pix2k = coords[notnan]
        
        rot= [0, 90]
        xsize= 5000

    else:
        raise NotImplementedError
    return {'name':name, 'coords': pix2k, 'Nresol': Nresol, 'rot': rot, 'xsize': xsize}


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
            assert len(mtup[1])==hp.nside2npix(Nsideresol) #ensures all maps are at Nside=2048
            assert len(mtup)==3 if self.with_sigma else 2 #ensures all maps either do or dont have uncertainties

    def get_sfd_offset_noise_for_patches(self, testbed_set):
        '''
        :param testbed_set: Output of eg: get_testbeds_latitudewise
        For each patch (typically an Nside=32 pixel) in the region, calculates the mean and std {recon[pix] - sfd[pix]}
            where pix is the set of Nside=2048 pixels lying in that patch
        :return: List of length number of maps
            output[im] = (mname, resdict) for the imth map
                resdict: 'set_name': Name of the region (collection of Nside=32 pixels)
                        'offset_mean': Length: Number of patches in region. Mean of the pixelwise offsets in each patch
                        'offset_std': Length: Number of patches in region. Std of the pixelwise offsets in each patch
        '''
        mapwise_sfd_offsets = [] #list of length len(self.maps)

        for mtuple in self.maps:
            if self.with_sigma:
                mname, meanmap, sigmamap = mtuple
            else:
                mname, meanmap = mtuple
            print(mname)
            resdict = {'set_name': testbed_set['set_name']} #separate result for each map

            patches = testbed_set['patches'] #list
            print(testbed_set['set_name'], len(patches))
            offset_mean, offset_std = [], []
            for patch in patches: #patch = Nside=32 pixel in the testbed
                mean, std = get_sfd_error(meanmap, self.sfdmap, patch['coords'], printout=False)
                offset_mean.append(mean)
                offset_std.append(std)
            resdict.update({'Offset_mean': np.array(offset_mean), 'Offset_std': np.array(offset_std)})
            mapwise_sfd_offsets.append((mname, resdict))
        return mapwise_sfd_offsets

    def get_sfd_offset_noise_for_patches_combined(self, testbed_set):
        '''
        :param testbed_set: Output of eg: get_testbeds_latitudewise
        For each patch (typically an Nside=32 pixel) in the region, calculates the mean and std {recon[pix] - sfd[pix]}
            where pix is the set of Nside=2048 pixels lying in that patch
        :return: NOTE: DIFFERENT OUTPUT FORMAT FROM ABOVE.
        Dictionary with:
            'set_name': Name of the region (collection of Nside=32 pixels)
            'mapwise_offset': List of length number of maps
                output['mapwise_offset'][m]: is a tuple:
                        0: Mpaname
                        1: Array of length |combined_pixels| and map[m][pix] - sfdmap[pix]
        '''
        mapwise_sfd_offsets = []  # list of length len(self.maps)
        resdict = {'set_name': testbed_set['set_name']}  #single dictionary output
        #extract all the pixels for the patches contained in region
        patches = testbed_set['patches']  # list
        print(testbed_set['set_name'], len(patches))
        combined_pixels = []

        for patch in patches:  # patch = Nside=32 pixel in the testbed
            combined_pixels.append(patch['coords'])
        combined_pixels = np.hstack(combined_pixels)

        for mtuple in self.maps:
            if self.with_sigma:
                mname, meanmap, sigmamap = mtuple
            else:
                mname, meanmap = mtuple
            print(mname)

            mapwise_sfd_offsets.append((mname, meanmap[combined_pixels] - self.sfdmap[combined_pixels]))
        resdict.update({'mapwise_offset': mapwise_sfd_offsets})
        return resdict

    def get_z_scores_for_patches(self, testbed_set, combined=False):
        '''
        :param testbed_set: Output of eg: get_testbeds_latitudewise
        :returns: the zscore of the patch relative to SFD = (recon_mean - sfd)/sigma
        ouput: List of length NMaps
                  output[m]: (mname, dict) for the mth map
                    dict['z-scores']: array with the zscores of all the Nside=2048 pixels in that region COMBINED if combined=True
                                    else, list of length Numpatches with each patch's pixels zscores in each element
        '''
        assert self.with_sigma
        mapwise_zscores = []

        for mname, meanmap, sigmamap in self.maps:
            print(mname)
            resdict = {'set_name': testbed_set['set_name']} #separate result for each map
            zscores_patches = []
            nanrecon = []
            for patch in testbed_set['patches']:
                z_score = (meanmap[patch['coords']] - self.sfdmap[patch['coords']])/sigmamap[patch['coords']]
                nanrecon.append(np.isnan(meanmap[patch['coords']]))
                zscores_patches.append(z_score)
            if combined:
                zscores_patches = np.hstack(zscores_patches)
                nanrecon = np.hstack(nanrecon)
                if np.sum(nanrecon)>0:
                    zscores_patches = zscores_patches[~nanrecon]
                    print('Nan in {} pixels'.format(np.sum(nanrecon)))
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
        plot_z_scores_vs_region(zscores, {})
    '''

    b17map, b17sigmamap = get_bayestar2017_map(return_sigma=True)
    recondict = pickle.load(open('../reconmaps/16a_fwhm-6-1__bgt21.pkl', 'rb'))
    recon16a = recondict['dustmap']
    sigmamap = np.sqrt(recondict['variancemap'])  # check if issues here

    compmaps = [('B17', b17map, b17sigmamap), ('FWHM=6.1\'', recon16a, sigmamap)]

    mapcomp = MapComparisons(compmaps)
    latrange = np.arange(30, 100, 10).astype('int')
    latrange = np.insert(latrange, 0, 25)
    latwise_offsets = []
    for lat in latrange:
        patches = get_testbeds_latitudewise(lat, get_pixels_in_Bayestar_footprint, Numoutput='max', Nresol=2048)
        offsets = mapcomp.get_sfd_offset_noise_for_patches_combined(patches)
        plot_noise_vs_region(offsets, {'dpi': 100, 'bins': 30})