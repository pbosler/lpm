#!/bin/bash

buildDir=$HOME/lpm/build-dfs-bve-paper
exe=$buildDir/examples/dfs_bve_rh54

initDepth=6
nlon=60
dt=0.02
tfinal=4.0

outputFilePrefix=/data/pabosle/dfs-bve-paper/rh54_zeta_Lambda_

time $exe -d $initDepth -nlon $nlon -dt $dt -tf $tfinal -o $outputFilePrefix 2>&1 | tee dfs_bve_rh54_out.txt

# with d = 6, m = 60, dt = 0.02 and tfinal = 4.0, the computation takes about 
# 8 minutes using 42 openmp threads on s1024454