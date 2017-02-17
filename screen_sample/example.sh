module load schrodinger/2016u3


echo "STEP 1 - MANUALLY ADDING STRUCTURE PROPERTIES"
echo "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~"
# The original structures, exported from the Schrödinger GUI,  are in
# `step-0_mae-original`. Copy them to `step-1_mae-manual` and manually add
# some structure properties to the *.mae files.
#
# These properties include:
#  * s_cs_pattern
#  * b_cs_first_match_only
#  * b_cs_use_substructure
#  * b_cs_both_enantiomers
#
# You can compare the differences between these *.mae files to see what's
# necessary for this example. Note that for `rh-hydrogenation-enamides.mae`,
# `b_cs_use_substructure` isn't included. If it's not included, it defaults to
# `False`. Similarly, `b_cs_both_enantiomers` is excluded from every *.mae
# except for `rh-hydrogenation-enamides.mae`. In
# `rh-hydrogenation-enamides.mae`, it's `True`, so the other enantiomer will be
# generated and used. For all the other structuers, `b_cs_both_enantiomers`
# defaults to False.

echo
echo "STEP 2 - AUTOMATICALLY ADDING ATOM AND BOND PROPERTIES"
echo "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~"
# Setup the atom and bond properties from the *.com files. These *.com files can
# be setup using the Schrödinger GUI.
python ../screen/setup_mae_from_com.py step-2_mae-automatic/binap.com step-1_mae-manual/binap.mae step-2_mae-automatic/binap.mae
python ../screen/setup_mae_from_com.py step-2_mae-automatic/binap-b.com step-1_mae-manual/binap-b.mae step-2_mae-automatic/binap-b.mae
python ../screen/setup_mae_from_com.py step-2_mae-automatic/s1.com step-1_mae-manual/s1.mae step-2_mae-automatic/s1.mae
python ../screen/setup_mae_from_com.py step-2_mae-automatic/rh-hydrogenation-enamides.com step-1_mae-manual/rh-hydrogenation-enamides.mae step-2_mae-automatic/rh-hydrogenation-enamides.mae

echo
echo "STEP 3 - MERGING STRUCTURES"
echo "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~"
# Merge the structures.
python ../screen/merge.py -g step-2_mae-automatic/rh-hydrogenation-enamides.mae -g step-2_mae-automatic/binap.mae step-2_mae-automatic/binap-b.mae -g step-2_mae-automatic/s1.mae -o step-3_merge-output-in-one.mae -d step-3_merge-output -m

echo
echo "STEP 4 - SETTING UP CONFORMATIONAL SEARCH FILES"
echo "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~"
# Setup the new *.com files. I'm going to copy the *.mae files over to a new
# directory before doing this in order to keep things clear.
cp step-3_merge-output/* step-4_conformational-search
# Here's how to do it one structure at a time.
cd step-4_conformational-search
python ../../screen/setup_com_from_mae.py rh-hydrogenation-enamides_binap-b_s1_5s.mae rh-hydrogenation-enamides_binap-b_s1_5s_single.com rh-hydrogenation-enamides_binap-b_s1_5s_single.mae
# This sets up all the *.mae files in one directory at the same time.
python ../../screen/setup_com_from_mae_many.py *.mae
cd ..
