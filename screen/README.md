# Automated Virtual Screening

Let's come up with a cooler name folks. This is so generic. I mean, Q2MM is so
catchy, I just feel like this also deserves some love.

The virtual screening software enables you to rapidly setup and start large
scale sampling of transition states when combined with a Q2MM TSFF.

Below are step by step instructions on how to use this software. You can follow
along with these steps with the example files inside the
[`screen_sample`](../screen_sample) directory.

---

### <center>Instructions</center>

#### 1. Manually creating and adding structure properties

If you want to use structures that aren't already added to the ligands,
substrates or reactions libraries, you will have to generate a Schrödinger
\*.mae structure file. You can do so in a number of ways. The simplest way is
probably to use the Maestro GUI.

You can clean up these \*.mae files by using one of the cleaning scripts in the
tools directory. Sometimes they contain many unnecessary properties. At any
rate, this is purely a question of asthetics.

You will then have to manually add several structure properties to the \*.mae
files. Again, these should already be entered into the \*.mae files inside the
existing libraries.

If any of these properties are missing, it will often attempt to use the default
values. Such is not the case for all properties, such as the SMARTS-esque
pattern which must be set by the user.

##### Structure properties

* `s_cs_pattern` - This is a SMARTS-eqsue pattern that is used to superimpose
structures. The code will go ahead and use any string property with "pattern" in
the name, so you can include multiple pattern properties.

   Ex.) `P-[Rh]-P` `O=C-N-C=C`

* `b_cs_first_match_only` - Sometimes the pattern identified can be matched in
more than one way. For example, `P-[Rh]-P` can match forwards and backwards,
resulting in the atoms, as an example, 2-1-3 or 3-1-2.

   If this value is true, then only the first match identified will be used.

   Default is false.

* `b_cs_use_substructure` - If true, use
`schrodinger.structutils.analyze.evaluate_substructure` to find the atom
indices for the SMILES-like pattern you provide. If false, it uses
`schrodinger.structutils.analyze.evaluate_smarts`.

   Default is false.

* `b_cs_both_enantiomers` - If true, the code will generate the opposite
enantiomer of the structure by inverting all of the x coordinates and then use
both the original structure and the other enantiomer. If false, just use the
original structure.

   Default is false.

#### 2. Automatically add atom and bond properties

For each \*.mae file, you need to setup the conformational search settings. The
easiest way to do so is to use the Maestro GUI. These settings can then be saved
to a \*.com file.

With the \*.mae and \*.com file in hand,
[`setup_mae_from_com`](../screen/setup_mae_from_com.py) can be used to copy the
necessary properties from the \*.com into the \*.mae.

##### Atom properties

* `b_cs_comp` - Correlates to the MacroModel COMP command. These atoms are
used for comparison of duplicate structures. Typically hydrogens aren't
included. True (`1`) includes the atom and false (`0`) excludes the atom.

* `b_cs_chig` - Correlates to the MacroModel CHIG command. `1` indicates a
chiral atom whose chirality should remain the same throughout the conformational
search. `0` indicates a non-chiral atom or an atom whose chirality may change
during the conformational search.

##### Bond properties

* `b_cs_tors` - Correlates to the MacroModel TORS command. `1` indicates a bond
to rotate during the conformational search, whereas `0` indicates no rotation
about that bond during the conformational search.

* `i_cs_rca4_1` - Correlates to the MacroModel RCA4 command. These are given in
sets of 4 atoms and identify ring breaks that should be made during the course
of the sampling. Say you you identify the dihedral 1-2-3-4. This would break the
bond 2-3 during the sampling. In this case, `b_cs_rca4_1` would be `1` and
`b_cs_rca4_2` would be `4`.

* `i_cs_rca4_2` - See `i_cs_rca4_1`.

* `i_cs_torc_a1` - Like `i_cs_rca_1` except for the TORC command.

* `i_cs_torc_a4` - Like `i_cs_rca_2` except for the TORC command and more
appropriately named.

* `i_cs_torc_a5` - Absolute value of the minimum torsional value allowed.

* `i_cs_torc_a6` - Absolute value of the maximum torsional value allowed.

* `i_cs_torc_b1` - Same as `i_cs_torc_a1`, but allows for a second TORC to be
defined.

* `i_cs_torc_b4` - Same as `i_cs_torc_a4`, but allows for a second TORC to be
defined.

* `i_cs_torc_b5` - Same as `i_cs_torc_a5`, but allows for a second TORC to be
defined.

* `i_cs_torc_b6` - Same as `i_cs_torc_a6, but allows for a second TORC to be
defined.

##### Usage

```
python setup_mae_from_com.py -h
python setup_mae_from_com.py comfile.com originalmaefile.mae outputmaefile.mae
```

Setting up the \*.com files for substrates or ligands is generally quite simple.
The automatic setup from MacroModel typically does the trick. However, apply
more thought and caution when setting up reaction templates. A lot of the
torsions, ring breaks and comparison atoms will have to be modified or at least
double checked manually.


#### 3. Merging structures

You're going to use
[`merge.py`](../screen/merge.py)
to combine structures. Look at that script's help
and documentation for more information.
[`merge.py`](../screen/merge.py)
uses information inside the
\*.mae files to determine how to superposition structures and delete duplicate
atoms.

Here's an example.

```
python merge.py -g reactions/rh_hydrogenation_enamides.mae
-g substrates/*.mae -g ligands/*.mae
```

This will take the structures from `substrates/*.mae` and add them on top of
`reactions/rh_hydrogenation_enamides.mae`. Then it will take all
those combined structures and add all the structures from
`ligands/*.mae` on top of them. You can add as many structures as
you'd like this way.

Here's another example.

```
python merge.py -g base1.mae base2.mae -g next1.mae -g another1.mae
another2.mae -g evenmore1.mae
```

This would do the following.

Of course, it would load the first group.

```
base1.mae
base2.mae
```

Then it would merge the second group.

```
base1-next1.mae
base2-next1.mae
```

Add the next group.

```
base1-next1-another1.mae
base1-next1-another2.mae
base2-next1-another1.mae
base2-next1-another2.mae
```

Merge the last group.

```
base1-next1-another1-evenmore1.mae
base1-next1-another2-evenmore1.mae
base2-next1-another1-evenmore1.mae
base2-next1-another2-evenmore1.mae
```

In order to make this work, the \*.mae structures that are being added need
to have certain properties. You can use `s_cs_pattern` or `s_cs_pattern1` or
`s_cs_pattern2` as a few examples. See the properties descriptions above.

It will search for these atoms in both structures, superposition those atoms
as best as Schrodinger can, and then merge the two structures. If properties
exist on the new atoms, those from the structure listed last by the `-g`
grouping, it will attempt to copy them over. The merging algorithm was actually
not so simple to write, so reference [`merge`](../screen/merge.py) for more
information.

NOTE: Substrates typically change more than ligands when adjusting to the TS
geometry. Given that is the case, it's wise to ADD THE SUBSTRATES to the
reaction template BEFORE THE LIGANDS in order to minimize the potential overlap
between structures.

#### 4. Setting up the conformational search files

All of the information inside the \*.mae files allows for
[`setup_com_from_mae`](../screen/setup_com_from_mae.py)
to automatically generate \*.com files to do sampling on the merged structure.

```
python setup_com_from_mae.py mergedfile.mae newcom.com outputmae.mae
```

You can also setup the \*.com files for many \*.mae files simultaneously using
[`setup_com_from_mae_many`](../screen/setup_com_from_mae_many.py).

```
python setup_com_from_mae_many.py somedir/*.mae
```

Say you had the files `maefilea.mae` and `maefileb.mae`, this would generate
`maefilea_cs.com` and `maefileb_cs.com`. Run these using MacroModel.

```
bmin -WAIT maefilea_cs
bmin -WAIT maefileb_cs
```

Running these would produce `maefilea_cs.mae` and `maefileb_cs.mae`.

There are also options inside `tools/setup_sh_from_com.py` that help with job
submission on the ND CRC.

#### 5. Setting up the redundant conformer elimination

This is very similar to step 4.

```
python setup_com_from_mae_many.py somedir/*.mae -j re
```

Add that jobtype "re" makes it do a redundant conformer elimination rather than
a conformational search. This should always be done following a conformational
search because, for some unknown reason, the energies reported in the
conformational search output (\*.log and \*.mae) are not reproducible.
