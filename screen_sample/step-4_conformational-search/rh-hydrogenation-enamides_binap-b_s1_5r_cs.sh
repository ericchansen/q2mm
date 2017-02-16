#!/bin/csh
#$ -M ericchansen@gmail.com
#$ -m ae
#$ -q debug
#$ -r n
#$ -N T1702152009_rh-hydrogenation-enamides_binap-b_s1_5r_cs

module load schrodinger/2016u3
setenv SCHRODINGER_TEMP_PROJECT /scratch365/ehansen3/.schrodtmp
setenv SCHRODINGER_TMPDIR /scratch365/ehansen3/.schrodtmp
setenv SCHRODINGER_JOBDB2 /scratch365/ehansen3/.schrodtmp

bmin -WAIT rh-hydrogenation-enamides_binap-b_s1_5r_cs
