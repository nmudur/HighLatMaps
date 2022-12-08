## Written by Core Francisco Park (corefranciscopark@g.harvard.edu cfpark00@gmail.com)


import numpy as np
import asdf
import os
import time
import torch
import pickle
import gc
import healpy as hp
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))

weights_dir=os.path.join(dir_path,"angular_cross_correlation/acc_temp_asdf")

### check if acc files have been pre-computed
assert len(glob.glob(os.path.join(weights_dir,"*")))>1, "Prepare acc files using angular_cross_correlation/Prepare_ACC.ipynb"

maskingnames=["NGC","FULL","South","North"]

with open(os.path.join(weights_dir,"info.pickle"),"rb") as f:
    info=pickle.load(f)

def get_acc_nside2048_batch(dustmaps_val,idustmapmask,n_bootstrap=None,device="cpu",dtype=torch.float64):
    """
    Evaluates the angular cross correlation with extragalactic sources.
    
    dustmaps_vals, (N_maps,n_pixels), where N_maps is the number of maps fed in and n_pixels are the values corresponsing to the nside=2048 indices in idustmapmask
    idustmapmask, (n_pixels), an array of nside=2048 (ring-ordered) indices, apply np.nonzero(mask)[0] to get this from a binary mask
    n_bootstrap int, number of bootstrap samples used, None uses the maximum amount(currently 10)
    device: device to compute on "cpu" recommended
    dtype: dtype used torch.float64 recommended
    """
    assert dustmaps_val.shape[1] == len(idustmapmask), "Dustmaps input should only have the selected pixels"
    if n_bootstrap is None:
        n_bootstrap=info["n_bootstrap"]
    assert info["n_bootstrap"]>=n_bootstrap, "Max bootstrap is"+str(info["n_bootstrap"])
    nside=2048
    npix=hp.nside2npix(nside)

    imask=np.load(os.path.join(weights_dir,"imask.npy"))
    
    commonmask=np.zeros(npix,np.uint8)
    commonmask[imask]+=1;commonmask[idustmapmask]+=1
    icommonmask=np.nonzero(commonmask==2)[0]
    
    buffer=np.full(npix,npix)
    uniques,indexes=np.unique(imask,return_index=True)
    buffer[uniques]=indexes
    iloading=buffer[icommonmask]
    
    buffer[...]=npix
    uniques,indexes=np.unique(idustmapmask,return_index=True)
    buffer[uniques]=indexes
    idustmap=buffer[icommonmask]
    del imask,commonmask,icommonmask,buffer,uniques,indexes
    gc.collect()
    
    n_maps=len(dustmaps_val)
    dustmaps_val_torch=torch.tensor(dustmaps_val[:,idustmap],device=device,dtype=dtype)
    dustmaps_masks=[]
    for i in range(n_maps):
        dustmaps_nans=torch.isnan(dustmaps_val_torch[i])
        dustmaps_masks.append(~dustmaps_nans)
        dustmaps_val_torch[i]-=dustmaps_val_torch[i][dustmaps_masks[i]].mean()
        dustmaps_val_torch[i][dustmaps_nans]=0
    dustmaps_masks=torch.stack(dustmaps_masks,dim=0).to(dtype=dtype)
    
    dustmaps_masks=dustmaps_masks.T
    dustmaps_val_torch=dustmaps_val_torch.T
    
    del dustmaps_nans,idustmap
    
    
    accs=[]
    for ib in range(n_bootstrap):
        for iz in range(info["nz"]):
            st=time.time()
            print("iz=",iz,"ib=",ib,end=" ")
            gc.collect()
            fn=os.path.join(weights_dir,"iz={}_ib={}.asdf".format(iz,ib))
            with asdf.open(fn) as f:
                weight=f.tree["weight"]
                weight=np.array(weight)
            weight=torch.tensor(weight[:,iloading],device=device,dtype=dtype)
            norms=weight@dustmaps_masks
            acc=(weight@dustmaps_val_torch)/norms
            accs.append(acc.cpu().detach().numpy())
            print(time.time()-st,"s")
    accs=np.stack(accs,axis=0).reshape(n_bootstrap,info["nz"],-1,n_maps).transpose(3,0,1,2)
    result={"accs":accs}
    result.update(info)
    result.update({"n_bootstrap":n_bootstrap})
    return result

def get_masking_indices(maskingname):
    fn=os.path.join(weights_dir,"intg_"+maskingname+"_ib={}.asdf".format(0))
    with asdf.open(fn) as f:
        indices=f.tree["indices"]
        indices=np.array(indices)
    return indices
    
def get_acc_intg_nside2048_batch(dustmaps_val,maskingname,n_bootstrap=None,device="cpu",dtype=torch.float64):
    """
    Evaluates the angular cross correlation with extragalactic sources.
    
    dustmaps_vals, (N_maps,n_pixels), where N_maps is the number of maps fed in and n_pixels are the values corresponsing to the nside=2048 indices in this masking name
    maskingname, str, name of mask to apply. One of ["NGC","FULL","South","North"]
    n_bootstrap int, number of bootstrap samples used, None uses the maximum amount(currently 10)
    device: device to compute on "cpu" recommended
    dtype: dtype used torch.float64 recommended
    
    returns: dict with accs: (Nmap, Nbootstrap, Nz)
    """
    assert maskingname in maskingnames, maskingname+" is not an option"
    if n_bootstrap is None:
        n_bootstrap=info["n_bootstrap"]
    assert info["n_bootstrap"]>=n_bootstrap, "Max bootstrap is"+str(info["n_bootstrap"])

    n_maps=len(dustmaps_val)
    assert np.isnan(dustmaps_val).sum()==0,"Nan found in map region"
    dustmaps_val_torch=torch.tensor(dustmaps_val,device=device,dtype=dtype)
    dustmaps_val_torch=dustmaps_val_torch-dustmaps_val_torch.mean(1,keepdim=True)
    dustmaps_val_torch=dustmaps_val_torch.T
    
    accs=[]
    for ib in range(n_bootstrap):
        print("ib=",ib,end=" ")
        st=time.time()
        gc.collect()
        fn=os.path.join(weights_dir,"intg_"+maskingname+"_ib={}.asdf".format(ib))
        with asdf.open(fn) as f:
            weight=f.tree["weight"]
            weight=np.array(weight)
        weight=torch.tensor(weight,device=device,dtype=dtype)
        acc=weight@dustmaps_val_torch
        accs.append(acc.cpu().detach().numpy())
        print(time.time()-st,"s")
    accs=np.stack(accs,axis=0).reshape(n_bootstrap,info["nz"],n_maps).transpose(2,0,1)
    result={"accs":accs}
    result.update(info)
    result.update({"n_bootstrap":n_bootstrap})
    return result

def get_acc_all_intg_nside2048_batch(dustmaps_val,maskingname,device="cpu",dtype=torch.float64):
    """
    Evaluates the angular cross correlation with extragalactic sources.
    
    dustmaps_vals, (N_maps,n_pixels), where N_maps is the number of maps fed in and n_pixels are the values corresponsing to the nside=2048 indices in this masking name
    maskingname, str, name of mask to apply. One of ["NGC","FULL","South","North"]
    device: device to compute on "cpu" recommended
    dtype: dtype used torch.float64 recommended
    """
    assert maskingname in maskingnames, maskingname+" is not an option"
    
    n_maps=len(dustmaps_val)
    assert np.isnan(dustmaps_val).sum()==0,"Nan found in map region"
    dustmaps_val_torch=torch.tensor(dustmaps_val,device=device,dtype=dtype)
    dustmaps_val_torch=dustmaps_val_torch-dustmaps_val_torch.mean(1,keepdim=True)
    dustmaps_val_torch=dustmaps_val_torch.T

    st=time.time()
    gc.collect()
    fn=os.path.join(weights_dir,"all_intg_"+maskingname+".asdf")
    with asdf.open(fn) as f:
        weight=f.tree["weight"]
        weight=np.array(weight)
    weight=torch.tensor(weight,device=device,dtype=dtype)
    acc=weight@dustmaps_val_torch
    accs=acc.cpu().detach().numpy().transpose(1,0)
    result={"accs":accs}
    result.update(info) #bug?
    return result

def get_acc_all_intg_nside2048_batch_debugmask(dustmaps_val,maskingname, iz, device="cpu",dtype=torch.float64):
    """
    Returns weight[iz, :]*dustmaps_val_torch (elemwise product): length Npix(2048)
    
    dustmaps_vals, (N_maps,n_pixels), where N_maps is the number of maps fed in and n_pixels are the values corresponsing to the nside=2048 indices in this masking name
    maskingname, str, name of mask to apply. One of ["NGC","FULL","South","North"]
    n_bootstrap int, number of bootstrap samples used, None uses the maximum amount(currently 10)
    device: device to compute on "cpu" recommended
    dtype: dtype used torch.float64 recommended
    
    """
    assert maskingname in maskingnames, maskingname+" is not an option"
    
    n_maps=len(dustmaps_val)
    assert np.isnan(dustmaps_val).sum()==0,"Nan found in map region"
    dustmaps_val_torch=torch.tensor(dustmaps_val,device=device,dtype=dtype)
    dustmaps_val_torch=dustmaps_val_torch-dustmaps_val_torch.mean(1,keepdim=True)
    dustmaps_val_torch=dustmaps_val_torch #(N_maps, npixels)

    st=time.time()
    gc.collect()
    fn=os.path.join(weights_dir,"all_intg_"+maskingname+".asdf")
    with asdf.open(fn) as f:
        weight=f.tree["weight"]
        weight=np.array(weight)
    weight_z=torch.tensor(weight,device=device,dtype=dtype)[iz, :].reshape((1, -1)) #(1, npixels)
    acc= torch.mul(weight_z, dustmaps_val_torch) #Nmaps, npixels
    accs=acc.cpu().detach().numpy().transpose(1,0)
    result={"accs":accs}
    result.update(info) #bug?
    return result

def get_rot_ind(nside,ang):
    assert ang in [0,90,180,270]
    npix=hp.nside2npix(nside)
    ipix=np.arange(npix)
    l,b=hp.pixelfunc.pix2ang(nside, ipix,lonlat=True)
    l=np.mod(l+ang,360)
    ipixres=hp.pixelfunc.ang2pix(nside,l,b,lonlat=True)
    return ipixres

def rotate_map(m,ang):
    nside=hp.npix2nside(len(m))
    l,b=hp.pix2ang(nside,np.arange(len(m)),lonlat=True)
    
    '''
    rotmaps=[]
    for ang in angs:
        print("ang",ang)
        l_rot=np.mod(l+ang,360)
        rotmaps.append(hp.pixelfunc.get_interp_val(m,l_rot,b,lonlat=True))
    '''
    print("ang", ang)
    l_rot = np.mod(l+ang,360)
    return hp.pixelfunc.get_interp_val(m,l_rot,b,lonlat=True) #)return np.stack(rotmaps,axis=0)

def smooth_rotate_map(dustmap,arcmins=[30],rots=[0,180]):
    nside=2048
    npix=hp.nside2npix(nside)
    assert len(dustmap)==npix,"use nside 2048 map"
    assert all([arcmin<=90 for arcmin in arcmins])
    assert all([rot in [0,90,180,270] for rot in rots])
    rotinds=[get_rot_ind(nside,ang=rot) for rot in rots]
    
    nanlocs=np.isnan(dustmap)
    dustmap[nanlocs]=dustmap[~nanlocs].mean()
    outmaps=[]
    for arcmin in arcmins:
        print("arcmin:",arcmin)
        dustmap_smooth=hp.smoothing(dustmap,sigma=np.radians(arcmin/60))
        dustmap_delta=dustmap-dustmap_smooth
        dustmap_delta[nanlocs]=np.nan
        for rot in rots:
            outmaps.append(rotate_map(dustmap_delta,rot))
    return np.stack(outmaps,axis=0).astype(np.float64)


## NM Additions
def preproc_get_acc_intgn(maps, names, reconpix, maskingname, smooths, savname, error_args, 
                         save_smoothed=True, save_acc_intgn=True):
    '''
    maps: [sfdmap, b17map, reconmap] list of maps wiht hp.nside2npix(2048) length
    reconpix: pixels at which the map is reconstructed
    maskingname = "NGC"
    smooths = [30]. No rots here. For evaluating the ACC of an input map, only 0 is needed. Fixed.
    savname:
    error_args: Ex:
        error_args = {'errortype':'Sampling', 'n_bootstrap': 10}
        error_args = {'rot_angles': np.linspace(30, 300, 100), 'errortype':'Rot', 'rot_batches': 10}
    
    returns: 
    List of accs for each input map. 
        if errortype=='Rot':
        (mapname, acc_intg_result for the input, mean_ensemble, median_ensemble, error_ensemble)
    Saves in a dictionary if save_acc_intgn=True
    
    '''
    assert all([len(m)==hp.nside2npix(2048) for m in maps])
    indices=get_masking_indices(maskingname)
    dustmaps_val = [] #delEmaps: has length N_sm*N_map (N_map in the usual single smooth case)
    names_aug = []
    for ifm,fullmap in enumerate(maps):
        dustmap=np.full(hp.nside2npix(2048),np.nan)
        dustmap[reconpix]=fullmap[reconpix]
        dustmaps_sm_rot= smooth_rotate_map(dustmap,arcmins=smooths,rots=[0]) #(N_sm, Npix2048)
        for ism, smooth in enumerate(smooths):
            dustmaps_val.append(dustmaps_sm_rot[ism, :])
            names_aug.append(names[ifm]+'_sm{}'.format(smooth))
            
    
    if not os.path.exists(savname[:savname.rindex('/')+1]):
        os.mkdir(savname[:savname.rindex('/')+1])
    
    if save_smoothed:
        pickle.dump(dustmaps_val, open(savname+'_smoothed.pkl', 'wb'))
    
    if error_args['errortype']=='Sampling': 
        accs_all = []
        for idx, dustmap in enumerate(dustmaps_val):
            accs=get_acc_intg_nside2048_batch(dustmap[indices].reshape((1, -1)), maskingname, n_bootstrap=error_args['n_bootstrap'], device="cpu")
            accs_intg=accs["accs"] #1, Nbts, Nz
            means=accs_intg.mean(1)
            stds=accs_intg.std(1,ddof=1)
            accs_all.append((names_aug[idx], accs, accs_intg, means, stds))
        if save_acc_intgn:
            savdict = {'accs_all':accs_all, 'names': names_aug, 'reconpix': reconpix, 'maskingname': maskingname, 'error_args': error_args, 'smooths': smooths}
            pickle.dump(savdict, open(savname+'_acc_intgn_bootstrapped.pkl', 'wb'))
            
    else: #errortype: Rotations
        accs_all = []
        for idx, dustmap in enumerate(dustmaps_val):
            assert len(dustmap.shape)==1
            accerrs_batch = []
            for b, rot_batch in enumerate(np.array_split(error_args['rot_angles'], error_args['rot_batches'])):
                rotatedmaps = np.stack([rotate_map(dustmap, err_angle) for err_angle in rot_batch]) #Error_angs, Npix
                accerrs_batch.append(get_acc_all_intg_nside2048_batch(rotatedmaps[:, indices], maskingname, device="cpu")) #acc: Error_angs, z
                del rotatedmaps
            accerrs_ensemble = np.vstack([elem['accs'] for elem in accerrs_batch])
            mean_ens, median_ens = np.mean(accerrs_ensemble, axis=0), np.median(accerrs_ensemble, axis=0) #mean should be 0?
            accres = get_acc_all_intg_nside2048_batch(dustmap[indices].reshape((1, -1)), maskingname, device="cpu")
            accs_all.append((names_aug[idx], accres, mean_ens, median_ens, accerrs_ensemble)) #edited
        if save_acc_intgn:
            savdict = {'accs_all':accs_all, 'names': names_aug, 'reconpix': reconpix, 'maskingname': maskingname, 'error_args': error_args, 'smooths': smooths}
            pickle.dump(savdict, open(savname+'_acc_intgn.pkl', 'wb'))
    return accs_all



def preproc_get_acc_intgn_debugmask(maps, names, reconpix, maskingname, smooths, savname, izarray,
                         save_smoothed=True, save_acc_intgn=True):
    '''
    maps: [sfdmap, b17map, reconmap] list of maps wiht hp.nside2npix(2048) length
    reconpix: pixels at which the map is reconstructed
    maskingname = "NGC"
    smooths = [30]. No rots here. For evaluating the ACC of an input map, only 0 is needed. Fixed.
    savname:
    izarray: which redshift INDICES to look at the mask at: each element in izarray is an index from zarray
    
    returns: 
    List of lists for each input map, iz
        if errortype=='Rot':
        (mapname, acc_intg_result for the input, mean_ensemble, median_ensemble, error_ensemble)
    Saves in a dictionary if save_acc_intgn=True
    
    '''
    assert all([len(m)==hp.nside2npix(2048) for m in maps])
    indices=get_masking_indices(maskingname)
    dustmaps_val = []
    names_aug = []
    for ifm,fullmap in enumerate(maps):
        dustmap=np.full(hp.nside2npix(2048),np.nan)
        dustmap[reconpix]=fullmap[reconpix]
        dustmaps_sm_rot= smooth_rotate_map(dustmap,arcmins=smooths,rots=[0]) #(N_sm, Npix2048)
        for ism, smooth in enumerate(smooths):
            dustmaps_val.append(dustmaps_sm_rot[ism, :])
            names_aug.append(names[ifm]+'_sm{}'.format(smooth))
            
    
    if not os.path.exists(savname[:savname.rindex('/')+1]):
        os.mkdir(savname[:savname.rindex('/')+1])
    
    if save_smoothed:
        pickle.dump(dustmaps_val, open(savname+'_smoothed.pkl', 'wb'))
    
    debugmask_mapwise = []
    for idx, dustmap in enumerate(dustmaps_val):
        assert len(dustmap.shape)==1
        '''
        accerrs_batch = []
        for b, rot_batch in enumerate(np.array_split(error_args['rot_angles'], error_args['rot_batches'])):
            rotatedmaps = np.stack([rotate_map(dustmap, err_angle) for err_angle in rot_batch]) #Error_angs, Npix
            accerrs_batch.append(get_acc_all_intg_nside2048_batch(rotatedmaps[:, indices], maskingname, device="cpu")) #acc: Error_angs, z
            del rotatedmaps
        '''
        zwise_weights = []
        for iz in izarray:
            zwise_weights.append(get_acc_all_intg_nside2048_batch_debugmask(dustmap[indices].reshape((1, -1)), maskingname, iz, device="cpu"))
        debugmask_mapwise.append((names_aug[idx], zwise_weights)) #edited
    if save_acc_intgn:
        savdict = {'debug_masks_mapwise': debugmask_mapwise, 'names': names_aug, 'reconpix': reconpix, 'maskingname': maskingname, 'smooths': smooths}
        pickle.dump(savdict, open(savname+'_debugmask.pkl', 'wb'))
    return debugmask_mapwise
