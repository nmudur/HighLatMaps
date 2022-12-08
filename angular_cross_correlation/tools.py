import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import healpy as hp
import scipy.spatial as sspat
import time
import sklearn.metrics as skmet
import torch

def subpix(nside,ipix,nsidesub):
    assert np.remainder(nsidesub,nside)==0 and np.remainder(np.log2(nsidesub//nside),1)==0
    fac=np.log2(nsidesub//nside)
    ipixnest=hp.ring2nest(nside,ipix)
    ipixnest=(ipixnest*(4**fac))[:,None]+np.arange(4**fac)[None,:]
    ipixnest=(ipixnest).astype(np.int64)
    return hp.nest2ring(nsidesub,ipixnest)

def ra_dec_to_l_b(ra,dec):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.value,c.galactic.b.value

def load_NGC(fp,ra="RA",dec="DEC",cuts={}):
    hdul=fits.open(fp)
    dat=hdul[1].data
    #print(dat.dtype)
    ls,bs=ra_dec_to_l_b(dat[ra],dat[dec])
    zs=dat.Z
    #sel=(bs>50)*(zs>0)
    sel=(zs>0)
    for cutkey,cut in cuts.items():
        sel*=cut(dat[cutkey])

    dat=np.stack([ls[sel],bs[sel],zs[sel]],axis=1)
    return dat

def get_acc_weighting(nside,sourcecoords,angbinedges,report_time=False,device="cpu"):
    if report_time:
        st=time.time()
    assert sourcecoords.l.unit==u.deg and sourcecoords.b.unit==u.deg and angbinedges.unit==u.deg,"all units in degrees please"
    source_l,source_b=sourcecoords.l.value,sourcecoords.b.value
    angbinedges=angbinedges.value
    n_binedges=len(angbinedges)
    maxang_rad=np.radians(np.max(angbinedges))
    angbinedges=torch.tensor(angbinedges,device=device)
    
    exgal_map=torch.zeros(n_binedges-1,hp.nside2npix(nside),device=device,dtype=torch.int64)
    
    for l,b in zip(source_l,source_b):

        vec=hp.pixelfunc.ang2vec(l,b, lonlat=True)

        ipix=hp.query_disc(nside,vec,maxang_rad,inclusive=True)

        l_pix,b_pix=hp.pixelfunc.pix2ang(nside, ipix,lonlat=True)

        ipix=torch.tensor(ipix,device=device)
        c=torch.deg2rad(torch.tensor(np.stack([b_pix,l_pix],axis=1),device=device))
        c0=torch.deg2rad(torch.tensor(np.array([b,l]),device=device))
        thetas=torch.rad2deg(2*torch.asin(torch.sqrt(torch.sin((c[:,0]-c0[0])/2)**2+torch.cos(c[:,0])*torch.cos(c0[0])*(torch.sin((c[:,1]-c0[1])/2)**2))))
        inds=torch.bucketize(thetas,angbinedges)
        notvalids=(inds==0)|(inds==n_binedges)
        inds-=1
        vals=torch.ones_like(inds,device=device,dtype=torch.int64)
        vals[notvalids]=0
        inds[notvalids]=0 #to avoid out of bound error
        exgal_map.index_put_((inds,ipix),vals,accumulate=True)

    if report_time:
        print("Time",time.time()-st)
    return exgal_map#,ts


def get_acc_intg_weighting(nside,sourcecoords,maxang,report_time=False,device="cpu"):
    if report_time:
        st=time.time()
    assert sourcecoords.l.unit==u.deg and sourcecoords.b.unit==u.deg and maxang.unit==u.deg,"all units in degrees please"
    source_l,source_b=sourcecoords.l.value,sourcecoords.b.value
    maxang_rad=np.radians(np.max(maxang.value))
    exgal_map=torch.zeros(hp.nside2npix(nside),device=device,dtype=torch.int64)
    
    for l,b in zip(source_l,source_b):

        vec=hp.pixelfunc.ang2vec(l,b, lonlat=True)

        ipix=hp.query_disc(nside,vec,maxang_rad,inclusive=True)

        l_pix,b_pix=hp.pixelfunc.pix2ang(nside, ipix,lonlat=True)

        ipix=torch.tensor(ipix,device=device)
        c=torch.deg2rad(torch.tensor(np.stack([b_pix,l_pix],axis=1),device=device))
        c0=torch.deg2rad(torch.tensor(np.array([b,l]),device=device))
        thetas=torch.rad2deg(2*torch.asin(torch.sqrt(torch.sin((c[:,0]-c0[0])/2)**2+torch.cos(c[:,0])*torch.cos(c0[0])*(torch.sin((c[:,1]-c0[1])/2)**2))))
        inds=torch.bucketize(thetas,angbinedges)
        notvalids=(inds==0)|(inds==n_binedges)
        inds-=1
        vals=torch.ones_like(inds,device=device,dtype=torch.int64)
        vals[notvalids]=0
        inds[notvalids]=0 #to avoid out of bound error
        exgal_map.index_put_((inds,ipix),vals,accumulate=True)

    if report_time:
        print("Time",time.time()-st)
    return exgal_map#,ts

###############################
#IGNORE
################










def get_acc_weighting2(nside,source_thetas,source_phis,theta,delta_theta,kernel_over_sigma=4,report_time=False):
    if report_time:
        st=time.time()
    npix=hp.nside2npix(nside)
    weighting=np.zeros(npix)
    
    
    theta_min,theta_max=np.clip(theta-(kernel_over_sigma*delta_theta),0,np.pi),np.clip(theta+(kernel_over_sigma*delta_theta),0,np.pi)
    thetas_ring,phis_ring=hp.pix2ang(nside,hp.query_strip(nside,theta_min,theta_max))
    weights=np.exp(-(thetas_ring-theta)**2/(2*(delta_theta**2)))
    weights/=weights.sum()

    rotmats=ang2rotmat(thetas=source_thetas,phis=source_phis)
    

    for rotmat in rotmats:
        thetas_ring_rot,phis_ring_rot=hp.rotator.rotateDirection(rotmat,thetas_ring,phis_ring)
        ipix,pixweights=hp.pixelfunc.get_interp_weights(nside, theta=thetas_ring_rot, phi=phis_ring_rot)
        np.add.at(weighting,ipix.flatten(),(pixweights*weights[None,:]).flatten())
    if report_time:
        print("Time",time.time()-st)
    return weighting




def get_acc_weighting3(nside,sourcecoords,maxang,report_time=False,device="cpu"):
    if report_time:
        st=time.time()
    assert sourcecoords.l.unit==u.deg and sourcecoords.b.unit==u.deg and maxang.unit==u.deg,"all units in degrees please"
    source_l,source_b=sourcecoords.l.value,sourcecoords.b.value
    maxang_rad=np.radians(maxang.value)
    exgal_map=torch.zeros(hp.nside2npix(nside),device=device,dtype=torch.float64)

    for l,b in zip(source_l,source_b):
        vec=hp.pixelfunc.ang2vec(l,b, lonlat=True)
        ipix=hp.query_disc(nside,vec,maxang_rad,inclusive=False)
        l_pix,b_pix=hp.pixelfunc.pix2ang(nside, ipix,lonlat=True)
        ipix=torch.tensor(ipix,device=device)
        c=torch.deg2rad(torch.tensor(np.stack([b_pix,l_pix],axis=1),device=device))
        c0=torch.deg2rad(torch.tensor(np.array([b,l]),device=device))
        thetas=torch.rad2deg(2*torch.asin(torch.sqrt(torch.sin((c[:,0]-c0[0])/2)**2+torch.cos(c[:,0])*torch.cos(c0[0])*(torch.sin((c[:,1]-c0[1])/2)**2))))
        r=torch.sin(thetas)
        w=1/(2*np.pi*r)
        #exgal_map[ipix]+=1
        exgal_map.index_put_((ipix,),w,accumulate=True)
    #ts=np.array(ts)
    if report_time:
        print("Time",time.time()-st)
    return exgal_map#,ts

def get_acc(m,sourcecoords,angbinedges,report_time=False,device="cpu"):
    if report_time:
        st=time.time()
    
    nside=hp.npix2nside(len(m))
    m=torch.tensor(m,device=device)
    assert sourcecoords.l.unit==u.deg and sourcecoords.b.unit==u.deg and angbinedges.unit==u.deg,"all units in degrees please"
    source_l,source_b=sourcecoords.l.value,sourcecoords.b.value
    angbinedges=angbinedges.value
    n_binedges=len(angbinedges)
    maxang_rad=np.radians(np.max(angbinedges))
    #print(np.max(angbinedges))
    angbinedges=torch.tensor(angbinedges,device=device)
    acc_values=torch.zeros(len(angbinedges)-1,dtype=torch.float64,device=device)
    counts=torch.zeros(len(angbinedges)-1,dtype=torch.int64,device=device)
    single=torch.ones(1,dtype=torch.int64,device=device)
    
    #ts=[]
    for l,b in zip(source_l,source_b):
        #t0=time.time()
        vec=hp.pixelfunc.ang2vec(l,b, lonlat=True)
        #t1=time.time()
        ipix=hp.query_disc(nside,vec,maxang_rad,inclusive=True)
        #t2=time.time()
        l_pix,b_pix=hp.pixelfunc.pix2ang(nside, ipix,lonlat=True)
        #t3=time.time()
        ipix=torch.tensor(ipix,device=device)
        vals=m[ipix]
        c=torch.deg2rad(torch.tensor(np.stack([b_pix,l_pix],axis=1),device=device))
        c0=torch.deg2rad(torch.tensor(np.array([b,l]),device=device))
        thetas=torch.rad2deg(2*torch.asin(torch.sqrt(torch.sin((c[:,0]-c0[0])/2)**2+torch.cos(c[:,0])*torch.cos(c0[0])*(torch.sin((c[:,1]-c0[1])/2)**2))))
        #print(torch.max(thetas))
        inds=torch.bucketize(thetas,angbinedges)
        valids=(inds!=0)&(inds!=n_binedges)
        inds=(inds[valids]-1,)
        acc_values.index_put_(inds,vals[valids],accumulate=True)
        counts.index_put_(inds,torch.ones_like(inds[0]),accumulate=True)
        #t4=time.time()
        #ts.append(np.array([t1-t0,t2-t1,t3-t2,t4-t3]))
    #ts=np.array(ts)
    if report_time:
        print("Time",time.time()-st)
    return acc_values,counts#,ts
