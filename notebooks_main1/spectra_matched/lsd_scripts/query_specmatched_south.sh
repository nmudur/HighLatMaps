#!/bin/bash
#SBATCH -J dr17_query
#SBATCH --account=finkbeiner_lab
#SBATCH -c 4
#SBATCH -t 0-01:00
#SBATCH -p test
#SBATCH --mem=40000
#SBATCH -o logs/specmatch_query_south_%j.o
#SBATCH -e logs/specmatch_query_south_%j.e

module load Anaconda/5.0.1-fasrc02
source activate LSD_env2
export OPTIONS="select _id, ra, dec, mag, mag_err, percentiles_E, percentiles_dm, gaia_id, l, b, plx, plx_err, chisq, gaia_edr3.source_id, gaia_edr3.l, gaia_edr3.b, gaia_edr3.pm, gaia_edr3.pmra, gaia_edr3.pmra_error, gaia_edr3.pmdec, gaia_edr3.pmdec_error, gaia_edr3.parallax, gaia_edr3.parallax_error, gaia_edr3.parallax_over_error, gaia_edr3.astrometric_n_good_obs_al, gaia_edr3.astrometric_chi2_al, gaia_edr3.astrometric_excess_noise, gaia_edr3.visibility_periods_used, gaia_edr3.phot_g_mean_mag, gaia_edr3.bp_rp, gaia_edr3.ruwe, gaia_dr2_source.l, gaia_dr2_source.b, gaia_dr2_source.ra, gaia_dr2_source.dec, gaia_dr2_source.ra_error, gaia_dr2_source.dec_error, gaia_dr2_source.solution_id, gaia_dr2_source.parallax, gaia_dr2_source.parallax_error, gaia_dr2_source.visibility_periods_used, gaia_dr2_source.astrometric_chi2_al, gaia_dr2_source.astrometric_n_good_obs_al, gaia_dr2_source.astrometric_excess_noise, gaia_dr2_source.phot_g_mean_mag, allwise.rchi2, allwise.w1rchi2, allwise.w1rchi2_pm, allwise.w1mpro, allwise.w1sigmpro, allwise.w2rchi2, allwise.w2mpro, allwise.w2sigmpro, allwise.w3rchi2, allwise.w3mpro, allwise.w3sigmpro, allwise.w4mpro, allwise.w4sigmpro, sdss_dr14_starsweep.psfflux, sdss_dr14_starsweep.psfflux_ivar, sdss_dr17_specobj.lsd_id, sdss_dr17_specobj.PLUG_RA, sdss_dr17_specobj.PLUG_DEC, sdss_dr17_specobj.CLASS, sdss_dr17_specobj.SUBCLASS, sdss_dr17_specobj.Z, sdss_dr17_specobj.Z_ERR, sdss_dr17_specobj.OBJID, sdss_dr17_specobj.ZWARNING, sdss_dr17_specobj.CHI68P, sdss_dr17_specobj.SN_MEDIAN_ALL, sdss_dr17_specobj.SN_MEDIAN, sdss_dr17_specobj.RCHI2 from bayestar19_stars, sdss_dr17_specobj(matchedto=bayestar19_stars, nmax=1, dmax=1), gaia_edr3(matchedto=bayestar19_stars, nmax=1, dmax=0.5), gaia_dr2_source(outer, matchedto=bayestar19_stars,nmax=1,dmax=0.5), allwise(outer, matchedto=bayestar19_stars, nmax=1, dmax=1), sdss_dr14_starsweep(outer, matchedto=bayestar19_stars, nmax=1, dmax=1)"

lsd-query --format=fits --output=spectrain_south_r50.fits --bounds='beam(0, -90, 50, coordsys="gal")' "$OPTIONS"
