import numpy as np
import pandas as pd
import joblib
import sklearn


def distmod_median_cut(df, mindm=8):
    return df['dm_median'].to_numpy() > mindm


def e_nonneg_cut(df):
    return df['E_median'].to_numpy() + df['E_sigma'].to_numpy() > 0


def e_nonneg_cut_sigfac(df, sigfac=1.0):
    return df['E_median'].to_numpy() + sigfac * df['E_sigma'].to_numpy() > 0


def e_sigma_cut(df, maxsig=0.3):
    return df['E_sigma'].to_numpy() < maxsig


def dm_sigma_cut(df, maxsig=1):
    return df['dm_sigma'].to_numpy() < maxsig


def rel_dm_cut(df, maxerr=1):
    return (df['dm_sigma'].to_numpy() / df['dm_median'].to_numpy()) < maxerr


def parallax_nan(df):
    return ~np.isnan(df['gaia_dr2_source.parallax'].to_numpy())


def parallax_nan_outer(df):
    return ~np.logical_or(np.isnan(df['gaia_dr2_source.parallax'].to_numpy()),
                          df['gaia_dr2_source.parallax'].to_numpy() == 0)


def highlat_cut(df):
    return ~((df['E_median'].to_numpy() > 1) * (df['E_sigma'].to_numpy() < 0.1))


def gaia_vispd(df, visthresh=8):
    return df['gaia_dr2_source.visibility_periods_used'].to_numpy() >= visthresh


def gaia_chisq(df):
    chi2thresh = 1.44 * np.maximum(np.exp(-0.4 * (df['gaia_dr2_source.phot_g_mean_mag'].to_numpy() - 19.5)), 1.0)
    return np.less((df['gaia_dr2_source.astrometric_chi2_al'].to_numpy() / (
                df['gaia_dr2_source.astrometric_n_good_obs_al'].to_numpy() - 5)), chi2thresh)


def bayestar_chisq(df, maxchisq=2.0):
    return df['reduced_chisq'].to_numpy() < maxchisq


def bayestar_chisq_per_passband(df, maxchisq=5.0):
    return df['chisq'].to_numpy() <= maxchisq


def pan_psf_ap_photdiff(df, thresh=0.1, numbands=2):
    return np.sum(df[df.columns[[c.startswith('ps.psf-apmag') for c in df.columns]]].to_numpy() < thresh,
                  axis=1) >= numbands


def pan_psf_ap_photdiff_cond1(df, thresh=0.1, errcut_frac=0.5, numbands=2):
    errcut = errcut_frac * thresh

    combvar = df[df.columns[[c.startswith('ps.psfmagstdev_') for c in df.columns]]].to_numpy() ** 2 + df[
        df.columns[[c.startswith('ps.apmagstdev_') for c in df.columns]]].to_numpy() ** 2
    # consider numbands with minimum combined error
    selbands = np.argsort(combvar, axis=1)[:, :numbands]
    minvar = np.take_along_axis(combvar, selbands, axis=1)
    psap_reliable = np.take_along_axis(df[df.columns[[c.startswith('ps.psf-apmag') for c in df.columns]]].to_numpy(),
                                       selbands, axis=1)

    maskerr = np.all(minvar < errcut ** 2, axis=1)
    galcut = np.all(psap_reliable > (0.1 - errcut), axis=1)
    return ~(galcut * maskerr)


def pan_psf_ap_photdiff_cond2(df, thresh=0.1, numbands=2, nsig=-1.0):
    combsig = np.sqrt(df[df.columns[[c.startswith('ps.psfmagstdev_') for c in df.columns]]].to_numpy() ** 2 + df[
        df.columns[[c.startswith('ps.apmagstdev_') for c in df.columns]]].to_numpy() ** 2)
    # consider numbands with minimum combined error
    selbands = np.argsort(combsig, axis=1)[:, :numbands]
    minsig = np.take_along_axis(combsig, selbands, axis=1)
    psap_reliable = np.take_along_axis(df[df.columns[[c.startswith('ps.psf-apmag') for c in df.columns]]].to_numpy(),
                                       selbands, axis=1)

    galcut = np.any((psap_reliable - nsig * minsig) > thresh,
                    axis=1)  # fixed on Jan 17, any runs prior to this had np.all
    return ~galcut


def panabs_psf_ap_photdiff_cond2(df, thresh=0.1, numbands=2, nsig=-1.0):
    combsig = np.sqrt(df[df.columns[[c.startswith('ps.psfmagstdev_') for c in df.columns]]].to_numpy() ** 2 + df[
        df.columns[[c.startswith('ps.apmagstdev_') for c in df.columns]]].to_numpy() ** 2)
    # consider numbands with minimum combined error
    selbands = np.argsort(combsig, axis=1)[:, :numbands]
    minsig = np.take_along_axis(combsig, selbands, axis=1)
    psap_reliable = np.take_along_axis(
        np.abs(df[df.columns[[c.startswith('ps.psf-apmag') for c in df.columns]]].to_numpy()), selbands, axis=1)

    galcut = np.any((psap_reliable - nsig * minsig) > thresh, axis=1)
    return ~galcut


def wise_svmnondetectioncombinedcut(df, model=None):
    if isinstance(model, str):
        model = joblib.load(model)
    assert len(model.coef_[0]) == 2, print('Code below assumes two input features')

    def svm_eqn(model, x_points):
        w = model.coef_[0]
        b = model.intercept_[0]
        return -(w[0] / w[1]) * x_points - b / w[1]

    star_svmval = df['z-W1'].to_numpy() - svm_eqn(model, df['r-i'].to_numpy())
    star_svmcut = (star_svmval < 0)
    star_svmcut = star_svmcut * (df['allwise.w1mpro'].to_numpy() != 0)
    wise_combined = np.logical_or(star_svmcut, df['allwise.w1mpro'].to_numpy() == 0)
    return wise_combined


def wise_svmnondetectioncombinedcut_nozreq(df, model=None):
    #more lenient, keeps the object if you have a nan in 3 ps1 bands
    if isinstance(model, str):
        model = joblib.load(model)
    assert len(model.coef_[0]) == 2, print('Code below assumes two input features')

    def svm_eqn(model, x_points):
        w = model.coef_[0]
        b = model.intercept_[0]
        return -(w[0] / w[1]) * x_points - b / w[1]

    star_svmval = df['z-W1'].to_numpy() - svm_eqn(model, df['r-i'].to_numpy())
    star_svmcut = (star_svmval < 0)
    star_svmcut = star_svmcut * (df['allwise.w1mpro'].to_numpy() != 0)
    nanps1 = np.isnan(df['mag_z'].to_numpy()) + np.isnan(df['mag_i'].to_numpy()) + np.isnan(df['mag_r'].to_numpy()) #True if any of the bands are nans
    wise_combined = np.logical_or(np.logical_or(star_svmcut, df['allwise.w1mpro'].to_numpy() == 0), nanps1)
    return wise_combined

def parallax_pm_combinedcut(df, parsigthresh, pmsigthresh):
    '''
    True if either
    np.abs(parallax) > parsigthresh*parallax_error (more than parsigthresh*sigma away from zero)
    pm > pmsigthresh*pm_error (greater than pmsigthresh)
    Larger sigthreshes are stricter cuts
    '''
    parmask = (np.abs(df['gaia_dr2_source.parallax'].to_numpy()) - parsigthresh * df[
        'gaia_dr2_source.parallax_error'].to_numpy()) > 0
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())

    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0

    return np.logical_or(parmask, pmmask)


def parallax_pm_combinedcut_and(df, parsigthresh, pmsigthresh, conditional=False):
    '''
    True if BOTH
    np.abs(parallax) > parsigthresh*parallax_error (more than parsigthresh*sigma away from zero)
    pm > pmsigthresh*pm_error (greater than pmsigthresh)
    Larger sigthreshes are stricter cuts
    '''
    parmask = (np.abs(df['gaia_dr2_source.parallax'].to_numpy()) - parsigthresh * df[
        'gaia_dr2_source.parallax_error'].to_numpy()) > 0
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())
    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0
    if not conditional:
        return np.logical_and(parmask, pmmask)
    if conditional:
        pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
        reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                          ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
        parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
        # True if either an 'unreliable' gaia reading OR is reliably bad
        return np.logical_and(parmask, pmmask)


def parallaxnan_conditional(df):
    nanplx = ~np.isnan(df['gaia_dr2_source.parallax'].to_numpy())
    reliableplx = np.logical_and(gaia_vispd(df), gaia_chisq(df))
    return nanplx * reliableplx


def pmcut(df, pmsigthresh):
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())
    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0
    return pmmask


def gaia_magcut(df, magcut=20):
    return df['gaia_dr2_source.phot_g_mean_mag'].to_numpy() < magcut


def w1_w2_w3cut(df, model):
    detected_all = (df['allwise.w1mpro'].to_numpy() != 0) * (df['allwise.w2mpro'].to_numpy() != 0) * (
                df['allwise.w3mpro'].to_numpy() != 0)
    dfdet = df.iloc[detected_all, :][['allwise.w1mpro', 'allwise.w2mpro', 'allwise.w3mpro']]
    w1w2, w2w3 = dfdet['allwise.w1mpro'].to_numpy() - dfdet['allwise.w2mpro'].to_numpy(), dfdet[
        'allwise.w2mpro'].to_numpy() - dfdet['allwise.w3mpro'].to_numpy()
    features = np.vstack([w1w2, w2w3]).T
    star_svmpred = model.predict(features)
    star_svmcut = star_svmpred > 0
    combinedcut = np.zeros(len(df), dtype=bool)
    combinedcut[~detected_all] = True
    combinedcut[detected_all] = star_svmcut
    return combinedcut

'''
#old: missing a *= in the second last and was allowing nans
def sdss_uvcut(df, model):
    detected_all = ~(np.isnan(df['sdss.pmag_u'].to_numpy()) + np.isnan(df['sdss.pmag_g'].to_numpy()) + np.isnan(
        df['sdss.pmag_r']).to_numpy())
    sdssmagcols = ['sdss.pmag_u', 'sdss.pmag_g', 'sdss.pmag_r']

    # ugr errs < 0.2
    sdss_errcols = df.columns[[c.startswith('sdss.pmag_err_') for c in df.columns]]
    sdss_pmag_err = df[sdss_errcols].to_numpy()
    detected_all *= np.all(sdss_pmag_err[:, :3] < 0.2, axis=1)
    print(np.sum(detected_all) / len(detected_all))

    # negative errors
    detected_all = ~np.any(sdss_pmag_err[:, :3] < 0, axis=1)

    # saturated on the faint end
    detected_all *= ~np.any(df[sdssmagcols].to_numpy() == 22.5, axis=1)  # nothing should have exactly 22.5

    if np.sum(detected_all) > 0:
        dfdet = df.iloc[detected_all, :]
        ug, gr = dfdet['sdss.pmag_u'].to_numpy() - dfdet['sdss.pmag_g'].to_numpy(), dfdet['sdss.pmag_g'].to_numpy() - \
                 dfdet['sdss.pmag_r'].to_numpy()
        features = np.vstack([ug, gr]).T
        star_svmpred = model.predict(features)
        star_svmcut = star_svmpred > 0
        combinedcut = np.zeros(len(df), dtype=bool)
        combinedcut[~detected_all] = True
        combinedcut[detected_all] = star_svmcut
        return combinedcut
    else:
        return ~detected_all
'''

def sdss_uvcut(df, model):
    # not nan in any of the bands
    detected_all = ~(np.isnan(df['sdss.pmag_u'].to_numpy()) + np.isnan(df['sdss.pmag_g'].to_numpy()) + np.isnan(
        df['sdss.pmag_r'].to_numpy()))
    sdssmagcols = ['sdss.pmag_u', 'sdss.pmag_g', 'sdss.pmag_r']

    # ugr errs < 0.2
    sdss_errcols = df.columns[[c.startswith('sdss.pmag_err_') for c in df.columns]]
    sdss_pmag_err = df[sdss_errcols].to_numpy()
    detected_all *= np.all(sdss_pmag_err[:, :3] < 0.2, axis=1)

    # no negative errors
    detected_all *= ~np.any(sdss_pmag_err[:, :3] < 0, axis=1)

    # saturated on the faint end
    detected_all *= ~np.any(df[sdssmagcols].to_numpy() == 22.5, axis=1)  # nothing should have exactly 22.5

    if np.sum(detected_all) > 0:
        dfdet = df.iloc[detected_all, :]
        ug, gr = dfdet['sdss.pmag_u'].to_numpy() - dfdet['sdss.pmag_g'].to_numpy(), dfdet['sdss.pmag_g'].to_numpy() - \
                 dfdet['sdss.pmag_r'].to_numpy()
        features = np.vstack([ug, gr]).T
        star_svmpred = model.predict(features)
        star_svmcut = star_svmpred > 0
        combinedcut = np.zeros(len(df), dtype=bool)
        combinedcut[~detected_all] = True
        combinedcut[detected_all] = star_svmcut
        return combinedcut
    else:
        return ~detected_all

    
    
def sdss_uvcut_strict(df, model):
    # not nan in any of the bands
    detected_all = ~(np.isnan(df['sdss.pmag_u'].to_numpy()) + np.isnan(df['sdss.pmag_g'].to_numpy()) + np.isnan(
        df['sdss.pmag_r'].to_numpy()))
    sdssmagcols = ['sdss.pmag_u', 'sdss.pmag_g', 'sdss.pmag_r']
    # saturated on the faint end: need this
    detected_all *= ~np.any(df[sdssmagcols].to_numpy() == 22.5, axis=1)  # nothing should have exactly 22.5
    # ugr errs < 0.2
    '''
    sdss_errcols = df.columns[[c.startswith('sdss.pmag_err_') for c in df.columns]]
    sdss_pmag_err = df[sdss_errcols].to_numpy()
    detected_all *= np.all(sdss_pmag_err[:, :3] < 0.2, axis=1)
    # no negative errors
    detected_all *= ~np.any(sdss_pmag_err[:, :3] < 0, axis=1)
    '''
        
    if np.sum(detected_all) > 0: #if there are any objects at all that pass detections
        dfdet = df.iloc[detected_all, :]
        ug, gr = dfdet['sdss.pmag_u'].to_numpy() - dfdet['sdss.pmag_g'].to_numpy(), dfdet['sdss.pmag_g'].to_numpy() - \
                 dfdet['sdss.pmag_r'].to_numpy()
        features = np.vstack([ug, gr]).T
        star_svmpred = model.predict(features)
        star_svmcut = star_svmpred > 0
        combinedcut = np.zeros(len(df), dtype=bool)
        combinedcut[~detected_all] = True
        combinedcut[detected_all] = star_svmcut
        return combinedcut
    else:
        return ~detected_all
    
def gaia_vispd_edr3(df):
    return df['gaia_edr3.visibility_periods_used'].to_numpy() >= 8


def gaia_ruwe_edr3(df):
    return df['gaia_edr3.ruwe'].to_numpy() < 1.4

def parallax_nan_edr3(df):
    return ~np.isnan(df['gaia_edr3.parallax'].to_numpy())

def parallaxcut(df, parsigthresh, dr=2, conditional=True):
    # earlier cuts included negative parallax
    if dr == 2:
        parmask = (df['gaia_dr2_source.parallax'].to_numpy() - parsigthresh * df[
            'gaia_dr2_source.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask

    if dr == 3:
        parmask = (df['gaia_edr3.parallax'].to_numpy() - parsigthresh * df['gaia_edr3.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd_edr3(df), gaia_ruwe_edr3(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask


def parallaxcut_abs(df, parsigthresh, dr=2, conditional=True):
    # earlier cuts included negative parallax
    if dr == 2:
        parmask = (np.abs(df['gaia_dr2_source.parallax'].to_numpy()) - parsigthresh * df[
            'gaia_dr2_source.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask

    if dr == 3:
        parmask = (np.abs(df['gaia_edr3.parallax'].to_numpy()) - parsigthresh * df['gaia_edr3.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd_edr3(df), gaia_ruwe_edr3(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask



def parallax_pm_combinedcut_or(df, parsigthresh, pmsigthresh, dr=2, conditional=True):
    # earlier cuts included negative parallax
    #pm * 𝜹pm = pmra * 𝜹pmra + pmdec * 𝜹pmdec
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())
    
    #select objects with pm>𝜹pm when pmsigthresh=1
    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0

    if dr == 2:
        #select objects with par>𝜹par when parsigthresh=1 (this version throws out all negative parallaxes)
        parmask = (df['gaia_dr2_source.parallax'].to_numpy() - parsigthresh * df[
            'gaia_dr2_source.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask + pmmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask + pmmask

    if dr == 3:
        parmask = (df['gaia_edr3.parallax'].to_numpy() - parsigthresh * df['gaia_edr3.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask + pmmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd_edr3(df), gaia_ruwe_edr3(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND has a parallax within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask + pmmask #Retain if it either has a high parallax OR a high proper motion

        
        
def parallax_pm_combinedcut_and_edr3(df, parsigthresh, pmsigthresh, dr=2, conditional=True):
    # earlier cuts included negative parallax
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())

    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0

    if dr == 2:
        parmask = (df['gaia_dr2_source.parallax'].to_numpy() - parsigthresh * df[
            'gaia_dr2_source.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask * pmmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask + pmmask

    if dr == 3:
        parmask = (df['gaia_edr3.parallax'].to_numpy() - parsigthresh * df['gaia_edr3.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask * pmmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd_edr3(df), gaia_ruwe_edr3(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            reliably_bad_pm = np.logical_and(pass_gaia_cuts, ~pmmask)
            pmmask = np.logical_or(pass_gaia_cuts, ~reliably_bad_pm) #ADDED THIS on 2/9
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask * pmmask
        

def parallax_pm_combinedcut_and_edr3_nonan(df, parsigthresh, pmsigthresh, dr=2, conditional=True):
    # earlier cuts included negative parallax
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())

    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0

    if dr == 2:
        parmask = (df['gaia_dr2_source.parallax'].to_numpy() - parsigthresh * df[
            'gaia_dr2_source.parallax_error'].to_numpy()) > 0

        if not conditional:
            return parmask * pmmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask * pmmask * (~np.isnan(df['gaia_dr2_source.parallax'].to_numpy()))

    if dr == 3:
        parmask = (df['gaia_edr3.parallax'].to_numpy() - parsigthresh * df['gaia_edr3.parallax_error'].to_numpy()) > 0
        
        if not conditional:
            return parmask * pmmask
        if conditional:
            pass_gaia_cuts = np.logical_and(gaia_vispd_edr3(df), gaia_ruwe_edr3(df))  # "reliable" gaia information
            reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                              ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
            parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
            reliably_bad_pm = np.logical_and(pass_gaia_cuts, ~pmmask)
            pmmask = np.logical_or(pass_gaia_cuts, ~reliably_bad_pm) #ADDED THIS on 2/9
            # True if either an 'unreliable' gaia reading OR is reliably bad
            return parmask * pmmask * (~np.isnan(df['gaia_edr3.parallax'].to_numpy()))

        
        
def parallax_pm_combinedcut_and_edr3_bayestar_nonan(df, parsigthresh, pmsigthresh, conditional=True):
    # earlier cuts included negative parallax
    pm_err = (1.0 / df['gaia_edr3.pm'].to_numpy()) * (
                df['gaia_edr3.pmra'].to_numpy() * df['gaia_edr3.pmra_error'].to_numpy() + df[
            'gaia_edr3.pmdec'].to_numpy() * df['gaia_edr3.pmdec_error'].to_numpy())

    pmmask = (df['gaia_edr3.pm'].to_numpy() - pmsigthresh * pm_err) > 0
    
    parmask = (df['plx'].to_numpy() - parsigthresh * df[
        'plx_err'].to_numpy()) > 0

    if not conditional:
        return parmask * pmmask * (~np.isnan(df['plx'].to_numpy()))
    if conditional:
        pass_gaia_cuts = np.logical_and(gaia_vispd(df), gaia_chisq(df))  # "reliable" gaia information
        reliably_bad_plx = np.logical_and(pass_gaia_cuts,
                                          ~parmask)  # True if it's a reliable gaia reading AND is within 1 sigma of 0
        parmask = np.logical_or(~pass_gaia_cuts, ~reliably_bad_plx)
        # True if either an 'unreliable' gaia reading OR is reliably bad
        return parmask * pmmask * (~np.isnan(df['plx'].to_numpy()))

def JK_2MASScut(df):
    jmag, kmag = df['mag_J'].to_numpy(), df['mag_K'].to_numpy()
    detected = ~(np.isnan(jmag) + np.isnan(kmag))
    
    return ((jmag - kmag)<1) + (~detected)
        
        
def wise_svmnondetectioncombinedcut_conditional(df, model=None):
    if isinstance(model, str):
        model = joblib.load(model)
    assert len(model.coef_[0]) == 2, print('Code below assumes two input features')

    def svm_eqn(model, x_points):
        w = model.coef_[0]
        b = model.intercept_[0]
        return -(w[0] / w[1]) * x_points - b / w[1]

    star_svmval = df['z-W1'].to_numpy() - svm_eqn(model, df['r-i'].to_numpy())
    star_svmcut = (star_svmval < 0)
    star_svmcut = star_svmcut * (df['allwise.w1mpro'].to_numpy() != 0)
    baddet = df['allwise.w1mpro'].to_numpy() == 0
    baddet += df['allwise.w1sigmpro'].to_numpy() < 0.2
    baddet += df['allwise.w1rchi2'].to_numpy() < 2
    wise_combined = np.logical_or(star_svmcut, baddet)
    return wise_combined