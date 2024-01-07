n_threads=4
cfs=$CFS_DIR'/bin/cfs'

$cfs -p propagation_dpdt.xml -t $n_threads propagation_dpdt 2>&1 | tee propagation_dpdt.log
