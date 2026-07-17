#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH -J freq
#SBATCH -o freq.output
#SBATCH -e freq.err
# Default in slurm
#SBATCH --mail-user mfarrugi@nd.edu
#SBATCH --mail-type=ALL
# Request 5 hours run time
#SBATCH -t 5:0:0
#SBATCH -p core 
#SBATCH -N 4
#
 
module load schrodinger

#home_dir = '/home/project/ff/specific_ff'

for index in {1..9}
do
    sed "s/xxxx/$index/g" freq.com  >  freq_$index.com

    $SCHRODINGER/bmin freq_$index
done

