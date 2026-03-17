#!/bin/csh
#$ -M youremail@domain.whatever
#$ -m ae
#$ -q long
#$ -r n
#$ -N T1703081440_job

module load schrodinger/2016u3
bmin -WAIT rh-hydrogenation-enamides_binap-b_s1_5r_cs_re
bmin -WAIT rh-hydrogenation-enamides_binap-b_s1_5s_cs_re
bmin -WAIT rh-hydrogenation-enamides_binap_s1_5r_cs_re
bmin -WAIT rh-hydrogenation-enamides_binap_s1_5s_cs_re
