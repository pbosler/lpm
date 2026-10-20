#!/bin/bash

buildDir=$HOME/lpm/build-dfs-bve-paper
exe=$buildDir/examples/dfs_bve_gauss_vort

outputFilePrefix=/data/pabosle/dfs-bve-paper/amr_example

circTol=0.2
initDepth=5
amrLimit=2

$exe -tf 0 -amr $amrLimit -d $initDepth -c $circTol -o $outputFilePrefix