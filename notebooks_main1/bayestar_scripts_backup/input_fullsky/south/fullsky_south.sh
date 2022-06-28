#!/bin/sh

#single.sh
INPFILENUM=$1
echo "Inputfilenum $INPFILENUM"
module load Anaconda/5.0.1-fasrc02
source activate pytorch_env2 

# Detect CentOS version number and CPU model
centos_ver=7
six_str=`grep "release 6" /etc/centos-release`
if [[ ! -z ${six_str} ]]; then
    centos_ver=6
fi

cpu_spec=`grep "model name" /proc/cpuinfo | head -n 1 `
cpu_flags=`grep "flags" /proc/cpuinfo | head -n 1 `

full_spec="${centos_ver} ${cpu_info} ${cpu_flags} ${bayestar_ver}"
full_hash=`echo ${full_spec} | md5sum | awk '{print $1}' | cut -c -8`

echo "CentOS ${centos_ver}"
echo "CPU spec: ${cpu_spec}"
echo "CPU flags: ${cpu_flags}"
grep "cache size" /proc/cpuinfo | head -n 1
grep "cpu MHz" /proc/cpuinfo | head -n 1
grep "cpu cores" /proc/cpuinfo | head -n 1
echo "Bayestar version: ${bayestar_ver}"
echo "Hash: ${full_hash}"

# On termination, propagate signal to entire process group
job_termination_handler()
{
    echo "+ Job termination handler:"
    echo "+   * Propagating signal to subprocesses ..."
    PGID=$(ps -o pgid= $$ | grep -o [0-9]*)
    kill -USR1 -$PGID
    echo "+   * Waiting 160 seconds ..."
    sleep 160
    echo "+   * Exiting ..."
    exit 17
}


trap 'job_termination_handler' USR1 # Job script should specify --signal=USR1@120

# Temporary working directory
work_dir=`env TMPDIR="/n/holyscratch01/finkbeiner_lab/Lab/nmudur" mktemp -d -p /n/holystratch01/finkbeiner_lab/Lab/nmudur/ -t bayestar.XXXXXXX`
echo "# Working directory: ${work_dir}"
cd ${work_dir}

export work_dir=${work_dir}

# Set up environment
if [[ ${centos_ver} -eq 6 ]]; then
    echo "Sourcing CentOS 6 environment ..."
    source /n/fink2/czucker/terra/activate-bayestar-centos6.sh
else
    echo "Sourcing CentOS 7 environment ..."
    source /n/fink2/czucker/terra/activate-bayestar-centos7.sh
fi

cp /n/home12/czucker/projects/bayestar_load_surfs.tar.gz ${work_dir}
cd ${work_dir}
tar -xzf bayestar_load_surfs.tar.gz
mv bayestar bayestar_load_surfs
cd bayestar_load_surfs
bash recompile.sh
cp bayestar ../
cd ${work_dir}

cp /n/fink2/czucker/terra/PS1_2MASS_Extinction_APOGEE.dat ${work_dir}
cp /n/fink2/czucker/terra/PS1_qz_2MASS_colors.dat ${work_dir}
cp /n/fink2/czucker/terra/PSMrLF.dat ${work_dir}

echo "Inputfilenum $INPFILENUM"
fn=$(printf "south.%05d.h5" ${INPFILENUM}) || exit 1 #EDIT
echo "Filenum $fn"
inputdir=/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/input_fullsky/south/ #EDIT
inputfile=${inputdir}/${fn}
gridfile=/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/output_fullsky/south/${fn} #EDIT
if test -f "$gridfile"; then
    echo "Output file already exists, either file mixup or previous run stopped midway?"
fi

bash /n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/controlscripts/process_file.sh ${work_dir}/bayestar ${inputfile} ${gridfile} /n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/config/it0/config.cfg &

pid=$!
wait ${pid}
    
echo "Done with ${fn}."
echo ""

#Make a new file to save the postprocessed output in
#mkdir ${postprocdir} Make it outside -- else each job would have created one
postprocfile=${inputdir}/postprocessed/tmpdir/${fn}
cp ${inputfile} ${postprocfile}

finalfile=${inputdir}/postprocessed/${fn}
if test -f "$finalfile"; then
    echo "Final PostProc file already exists, file mixup?"
    exit 1
fi



bash /n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/compute_append_gmm.sh ${gridfile} ${postprocfile} "delete" || exit 1
mv ${postprocfile} ${finalfile}


echo "Appending posterior means, sigmas to the input file"
# Delete working directory
echo "# Deleting the working directory ..."
rm -rf "${work_dir}"
