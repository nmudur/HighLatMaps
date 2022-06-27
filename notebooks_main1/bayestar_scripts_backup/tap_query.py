import pyvo as vo
import numpy as np
import astropy
import time
import os
import sys
import h5py
import numpy.ma as ma

from astropy.table import Table
import socket
import urllib3

#adapting code from https://colab.research.google.com/drive/1lPzhGSSIjx2nQ7XM2v8bQZtkf0Atrk0z?usp=sharing#scrollTo=3vsgwYHFdI63
def submit_vo_query(service, table_name, sid, columns=None):
    if columns is None:
        col_str = r'ext.*'
    else:
        col_str = ',\n          '.join(columns)
    qstr = f"""
        SELECT
          {col_str}
        FROM {table_name} AS ext
        JOIN TAP_UPLOAD.cat AS cat
        USING (source_id)"""
    input_cat = astropy.table.Table(
        data=[sid],
        names=['source_id'],
        dtype=['int64']
    ) #this is TAP_UPLOAD
    # print(input_cat)
    job = service.submit_job(qstr, uploads={'cat':input_cat})
    job.run()
    return job


def submit_batched_vo_queries(service, table_name, sid, batch_size,
                              columns=None, verbose=False):
    n = len(sid)
    n_batches = int(np.ceil(n/batch_size))
    job_list = []
    for k in range(n_batches):
        i0 = k * batch_size
        sid_batch = sid[i0:i0+batch_size]
        if verbose:
            print(f'Submitting job {k+1} of {n_batches} ...')
        job_list.append(submit_vo_query(
            service,
            table_name,
            sid_batch,
            columns=columns
        ))
    return job_list

def wait_for_jobs(job_list, verbose=False, wait=1):
    active_jobs = set()
    for k,j in enumerate(job_list):
        active_jobs.add((k,j))

    results = {}

    while active_jobs:
        for k,j in list(active_jobs):
            if j.phase not in ('QUEUED', 'EXECUTING'):
                active_jobs.remove((k,j))
                if verbose:
                    print(f'Job {k} {j.phase}. Fetching results ...')
                try:
                    j.raise_if_error()
                except vo.DALQueryError as err:
                    print(err.reason)
                    raise err
                results[k] = j.fetch_result()
                j.delete()
        if active_jobs:
            if verbose:
                print(f'{len(active_jobs)} jobs remaining ...')
            time.sleep(wait) 
    return [results[k] for k in range(len(job_list))]


def concat_results(result_list):
    return astropy.table.vstack([res.to_table() for res in result_list])


def query_workflow(service, table_name, sid,
                   batch_size=100000, columns=None,
                   wait=5, verbose=False):
    job_list = submit_batched_vo_queries(
        service, table_name, sid,
        batch_size=batch_size, columns=columns, verbose=verbose
    )
    result_list = wait_for_jobs(job_list, wait=wait, verbose=verbose)
    result_table = concat_results(result_list)
    return result_table

def query_fidelity(sid, verbose=True):
    #sid: Source Id List
    fid_service = vo.dal.TAPService('http://dc.zah.uni-heidelberg.de/tap')
    fields = ['source_id', 'fidelity_v2']
    fid_data = query_workflow(
            fid_service,
            'gedr3spur.main',
            sid,
            verbose=verbose,
            columns=fields,
            batch_size=20000)
    return fid_data


def query_fidelity_short(sid):
    fid_service = vo.dal.TAPService('http://dc.zah.uni-heidelberg.de/tap')
    fields = ['source_id', 'fidelity_v2']
    queryjob = submit_vo_query(fid_service,'gedr3spur.main', sid, columns=fields)
    print(queryjob.phase)
    while queryjob.phase in ('QUEUED', 'EXECUTING'):
        print('Job Status:', queryjob.phase)
        time.sleep(1)
    print(queryjob.phase)
    result = queryjob.fetch_result()
    return result.to_table()


def test():
    #load from npy and compare with GAVO query to check if same
    npydata = np.load('../fidelity/fidelities/lvl5_9.npy')
    sidlist = np.array([tup[0] for tup in npydata])
    fidlist = np.array([tup[1] for tup in npydata])
    
    #Long: Exactly like the colab notebook:
    querydata = query_fidelity(sidlist)
    fidquery = querydata['fidelity_v2']
    print(type(fidquery))
    fiddata = fidquery.data.filled(fill_value=np.nan)
    print(type(fiddata), 'Num nans', np.isnan(fiddata).sum())
    print(fiddata[:10])
    
    
    #Short: Without batching, also returns the same as the npy
    #querydata = query_fidelity_short(sidlist)
    print(len(sidlist))
    print(len(querydata), len(querydata[0]))
    print('Npy first ten')
    print(npydata[:10])
    print('GAVO query first ten')
    print(querydata[:10])
    return

def missing_srcs():
    print('Ok Srcs')
    sidlist = np.array([3766500127465575680, 3766500196185052800, 3766502253474425856,
 3766502287834163968, 3766502292129101568, 3766502322193903744,
 3766502665791376512, 3766502944964126720, 3766503078108239232,
 3766503116762815616])
    querytab = query_fidelity(sidlist)
    print(querytab)
    print('Missing Srcs') 
    sidlist = np.array([3766502360847911552, 3766502390913473280, 3766503357280854400, 5692532100587670016, 5692532134947702400, 5692544057776685696])
    querytab = query_fidelity(sidlist)
    print(querytab)
    return     
    
'''
def process_save_fidelity(filename):
    #Filename: this is the already-copied-into-input file that you need to grab the source ids of and query fidelity and fill
    # the pi, pi_err columns accordingly 
    modf = h5py.File(filename, 'a')
    for pixel in modf['photometry'].keys():
        try:
            source_ids = np.array(modf['photometry/{}'.format(pixel)]['gaia.source_id'])
            fidtable = query_fidelity(source_ids)
            print('Discrepancy', len(source_ids), np.sum(source_ids==0), len(fidtable))
            #debug lines
            missing_src = source_ids[~np.isin(source_ids, fidtable['source_id'].data.filled(fill_value=-999))]
            missfid = query_fidelity(missing_src)
            print('Missing', len(missing_src), np.unique(missing_src),  len(missfid))
            ok_src = source_ids[np.isin(source_ids, fidtable['source_id'].data.filled(fill_value=-999))]
            print('Ok Sources', len(ok_src), np.unique(ok_src)[:10])
            #debug lines end
            fidarr = fidtable['fidelity_v2'].data.filled(fill_value=np.nan)
            idx_gaia = fidarr>0.5
            print(type(fidarr))
            print(f'pixid {pixel}: Good:{np.sum(idx_gaia)}, All: {len(idx_gaia)}')
            print(len(source_ids), len(np.unique(source_ids)), len(fidtable['source_id'].data.filled(fill_value=-999)))
            assert np.allclose(source_ids, fidtable['source_id'].data.filled(fill_value=-999))
            modf['photometry/{}'.format(pixel)]['pi'][idx_gaia] = modf['photometry/{}'.format(pixel)]['gaia.parallax'][idx_gaia]
            modf['photometry/{}'.format(pixel)]['pi_err'][idx_gaia] = modf['photometry/{}'.format(pixel)]['gaia.parallax_error'][idx_gaia]
        except Exception as e:
            print(repr(e), filename, pixel)
    modf.close()
    return
'''

def process_save_fidelity(filename):
    #Filename: this is the already-copied-into-input file that you need to grab the source ids of and query fidelity and fill
    # the pi, pi_err columns accordingly
    # Handles missing sources 
    modf = h5py.File(filename, 'a')
    for pixel in modf['photometry'].keys():
        try:
            source_ids = np.array(modf['photometry/{}'.format(pixel)]['gaia.source_id'])
            fidtable = query_fidelity(source_ids)
            print('Discrepancy: Sources in input, and sources for which outputs are available', len(source_ids), np.sum(source_ids==0), len(fidtable))
            fidarr = fidtable['fidelity_v2'].data.filled(fill_value=np.nan)

            sidinfidoutmask = np.isin(source_ids, fidtable['source_id'].data.filled(fill_value=-999)) #Sources for which fidelity classifier returns an output
            print('Check no nans', np.isnan(fidarr).sum())
            fidall = np.zeros(len(source_ids))
            fidall[sidinfidoutmask] = fidarr
            idx_gaia = fidall>0.5
            print(f'pixid {pixel}: Good:{np.sum(idx_gaia)}, All: {len(idx_gaia)}')
            print('Sources, Unique, Table length', len(source_ids), len(np.unique(source_ids)), len(fidtable['source_id'].data.filled(fill_value=-999)))

            
            assert np.allclose(source_ids[sidinfidoutmask], fidtable['source_id'].data.filled(fill_value=-999)) #asserting that the order of sources in the returned table is the same as in the list input
            print('preassigni: length check', len(idx_gaia), len(modf['photometry/{}'.format(pixel)]['pi']))
            plxnew = np.array(modf['photometry/{}'.format(pixel)]['pi'])
            plxerrnew = np.array(modf['photometry/{}'.format(pixel)]['pi_err'])
            plxnew[idx_gaia] = modf['photometry/{}'.format(pixel)]['gaia.parallax'][idx_gaia]
            plxerrnew[idx_gaia] = modf['photometry/{}'.format(pixel)]['gaia.parallax_error'][idx_gaia]
            modf['photometry/{}'.format(pixel)]['pi'] = plxnew 
            modf['photometry/{}'.format(pixel)]['pi_err'] = plxerrnew 
            print(f'Assigned: idx_gaia= {np.sum(idx_gaia)}. Nonzero parallax= ', np.sum(plxnew!=0), np.sum(modf['photometry/{}'.format(pixel)]['pi']!=0), np.sum(modf['photometry/{}'.format(pixel)]['pi_err']!=1e10))
            print('##############################')
            print('{}: Parallax used for :{:.3f} % sources'.format(pixel, 100*np.sum(modf['photometry/{}'.format(pixel)]['pi_err']<1e6)/len(modf['photometry/{}'.format(pixel)]['pi_err']))) 
            print('##############################')

        except Exception as e:
            print(repr(e), filename, pixel)
    modf.close()
    return

def process_save_fidelity_faster(filename):
    # Filename: this is the already-copied-into-input file that you need to grab the source ids of and query fidelity and fill
    # Does a single query call for all pixels in the file, earlier you had a single query call per pixel
    # the pi, pi_err columns accordingly
    # Handles missing sources 
    modf = h5py.File(filename, 'a')
    source_ids, ruwes = [], []
    lower_idx = np.zeros(len(modf['photometry'].keys()), dtype=int)

    for ip, pixel in enumerate(modf['photometry'].keys()):
        source_ids.append(np.array(modf['photometry/{}'.format(pixel)]['gaia.source_id']))
        ruwes.append(np.array(modf['photometry/{}'.format(pixel)]['gaia.ruwe']))
        if ip<len(modf['photometry'].keys())-1:
            lower_idx[ip+1] = lower_idx[ip]+len(np.array(modf['photometry/{}'.format(pixel)]['gaia.source_id']))
    print(lower_idx)
    source_ids = np.hstack(source_ids)
    ruwes = np.hstack(ruwes)
    print('Pre query', flush=True)
    qcount = 0
    while qcount<20:
        try:
            fidtable = query_fidelity(source_ids)
        except (socket.timeout, socket.error, urllib3.exceptions.ReadTimeoutError, vo.dal.exceptions.DALServiceError) as e:
            print(f'Error {repr(e)} on Attempt {qcount}')
            qcount+=1
            time.sleep(60)
        else:
            qcount+=1
            break #if it goes all the way till the end without querying then fidtable wont exist and you'll get an error in subsequent lines
    print('Discrepancy: Sources in input, Sources in input with sid=0, and sources for which outputs are available', len(source_ids), np.sum(source_ids==0), len(fidtable))
    fidarr = fidtable['fidelity_v2'].data.filled(fill_value=np.nan)

    sidinfidoutmask = np.isin(source_ids, fidtable['source_id'].data.filled(fill_value=-999)) #Sources for which fidelity classifier returns an output
    print('Check no nans', np.isnan(fidarr).sum())
    fidall = np.zeros(len(source_ids))
    fidall[sidinfidoutmask] = fidarr
    idx_gaia = (fidall>0.5) * (ruwes<1.4)
    print(f'pixid {pixel}: Good:{np.sum(idx_gaia)}, All: {len(idx_gaia)}')
    print('Sources, Unique, Table length', len(source_ids), len(np.unique(source_ids)), len(fidtable['source_id'].data.filled(fill_value=-999)))


    assert np.allclose(source_ids[sidinfidoutmask], fidtable['source_id'].data.filled(fill_value=-999)) #asserting that the order of sources in the returned table is the same as in the list input

    for ip, pixel in enumerate(modf['photometry'].keys()):
        try:
            upper_idx = lower_idx[ip+1] if (ip<len(modf['photometry'].keys())-1) else len(idx_gaia)
            print(f'{ip}, {pixel}: low_idx={lower_idx[ip]}, upp_idx={upper_idx}')
            pix_criteria = idx_gaia[lower_idx[ip]:upper_idx]
            pix_fids = fidall[lower_idx[ip]:upper_idx]

            plxnew = np.array(modf['photometry/{}'.format(pixel)]['pi'])
            plxerrnew = np.array(modf['photometry/{}'.format(pixel)]['pi_err'])
            #print Preassigning Length Check
            assert len(pix_criteria) == len(plxnew)
            plxnew[pix_criteria] = modf['photometry/{}'.format(pixel)]['gaia.parallax'][pix_criteria]
            plxerrnew[pix_criteria] = modf['photometry/{}'.format(pixel)]['gaia.parallax_error'][pix_criteria]
            modf['photometry/{}'.format(pixel)]['pi'] = plxnew
            modf['photometry/{}'.format(pixel)]['pi_err'] = plxerrnew 
            modf['photometry/{}'.format(pixel)]['fidelity_v2'] = pix_fids
            print(f'Assigned: idx_gaia= {np.sum(idx_gaia)}. Nonzero parallax= ', np.sum(plxnew!=0), np.sum(modf['photometry/{}'.format(pixel)]['pi']!=0), np.sum(modf['photometry/{}'.format(pixel)]['pi_err']!=1e10))
            print('##############################')
            print('{}: Parallax used for :{:.3f} % sources'.format(pixel, 100*np.sum(modf['photometry/{}'.format(pixel)]['pi_err']<1e6)/len(modf['photometry/{}'.format(pixel)]['pi_err'])))
            print('##############################')

        except Exception as e:
            print(repr(e), filename, pixel)
            raise e
    modf.close()
    return
'''
def presave_fidelity_query(inputdir):
    fall = os.listdir(inputdir)
    hflist = []
    for f in fall:
        if f.endswith('.h5'):
            hflist.append(f)
    source_ids = []
    for hf in hflist:
        filename = os.path.join(inputdir, hf)
        modf = h5py.File(filename, 'r')
        for ip, pixel in enumerate(modf['photometry'].keys()):
            source_ids.append(np.array(modf['photometry/{}'.format(pixel)]['gaia.source_id']))
    source_ids = np.hstack(source_ids)
    uniquesid = np.unique(source_ids)
    fidtable = query_fidelity(uniquesid)
    print('Discrepancy: Sources in input, and sources for which outputs are available', len(source_ids), len(uniquesid), np.sum(uniquesid==0), len(fidtable))
    fidarr = fidtable['fidelity_v2'].data.filled(fill_value=np.nan) 
    sidinfidoutmask = np.isin(uniquesid, fidtable['source_id'].data.filled(fill_value=-999)) #Sources for which fidelity classifier returns an output
    print('Check no nans', np.isnan(fidarr).sum())
    fidall = np.zeros(len(uniquesid))
    fidall[sidinfidoutmask] = fidarr
    assert np.allclose(uniquesid[sidinfidoutmask], fidtable['source_id'].data.filled(fill_value=-999)) #asserting that the order of sources in the returned table is the same as in the list input
    print('Dtype:', uniquesid.dtype, fidall.dtype)
    
    return
'''

if __name__=='__main__':
    #test()
    #missing_srcs()
    #process_save_fidelity_faster('input_5-16/Test/stripes.00005_test.h5')
    process_save_fidelity_faster(sys.argv[1])
