n_threads=4
cfs=$CFS_DIR'/bin/cfs'

$cfs -p propagation_PCWE.xml -t $n_threads propagation_PCWE 2>&1 | tee propagation_PCWE.log
