# Automated Virtual Screening

Let's come up with a cooler name folks.

The virtual screening software enables you to rapidly setup and start
large scale sampling of TS's when combined with a Q2MM TSFF.

## Instructons

### 1. Manually creating and adding structure properties

If you want to use structures that aren't already added to the ligands,
substrates or reactions libraries, you will have to generate a Schrödinger .mae
structure file. You can do so in a number of ways. The simplest way is probably
to use the Maestro GUI.

You can clean up these *.mae files by using one of the cleaning scripts in the
tools directory. Sometimes they contain many unnecessary properties. At any
rate, this is purely a question of asthetics.

You will then have to manually add several structure properties to the *.mae
files. Again, these should already be entered into the *.mae files inside the
existing libraries.

##### Properties

If any of these properties are missing, it will often attempt to use the default
values. Such is not the case for all properties, such as the SMARTS-esque
pattern which must be set by the user.

* `s_cs_pattern` - This is a SMARTS-eqsue pattern that is used to superimpose
structures. The code will go ahead and use any string property with "pattern" in
the name.

Ex.) `P-[Rh]-P` `O=C-N-C=C`

* `b_cs_first_match_only` - Sometimes the pattern identified can be matched in
more than one way. For example, `P-[Rh]-P` can match forwards and backwards,
resulting in the atoms, as an example, 2-1-3 or 3-1-2.

If this value is `True`, then only the first match identified will be used.

Default is `False`.

* `b_cs_use_substructure` - If `True`,
`schrodinger.structutils.analyze.evaluate_substructure` to find the atom
indices for the SMILES-like pattern you provide. If `False`, it uses
`schrodinger.structutils.analyze.evluate_smarts`.

Default is `False`.

* `b_cs_both_enantiomers` - If `True`, the code will generate the opposite
enantiomer of the structure by inverting all of the x coordinates and then use
both the original structure and the other enantiomer. If `False`, just use the
original structure.

Default is `False`.

### 2. Automatically add atom and bond properties

For each *.mae file, you need to setup the conformational search settings. The
easiest way to do so is to use the Maestro GUI. These settings can then be saved
to a *.com file.

With the *.mae and *.com file in hand, `setup_mae_from_com` can be used to copy
the necessary properties from the *.com into the *.mae.

##### Atom properties

* `b_cs_comp` - Correlates to the MacroModel COMP command. These atoms are
used for comparison of duplicate structures. Typically hydrogens aren't
included. `1` includes the atom and `0` excludes the atom.

* `b_cs_chig` - Correlates to the MacroModel CHIG command. `1` indicates a
chiral atom whose chirality should remain the same throughout the conformational
search. `0` indicates a non-chiral atom or an atom whose chirality may change
during the conformational search.

##### Bond properties

* `b_cs_tors` - Correlates to the MacroModel TORS command. `1` indicates a bond
to rotate during the conformational search, whereas `0` indicates no rotation
about that bond during the conformational search.

* `b_cs_rca4_1` - Correlates to the MacroModel RCA4 command. These are given in
sets of 4 atoms and identify ring breaks that should be made during the course
of the sampling. Say you you identify the dihedral 1-2-3-4. This would break the
bond 2-3 during the sampling. In this case, `b_cs_rca4_1` would be `1` and
`b_cs_rca4_2` would be `4`.

* `b_cs_rca4_2` - See `b_cs_rca4_1`.

##### Usage

```
python setup_mae_from_com.py -h
python setup_mae_from_com.py comfile.com originalmaefile.mae outputmaefile.mae
```

Setting up the *.com files for substrates or ligands is generally quite simple.
The automatic setup from MacroModel typically does the trick. However, apply
more thought and caution when setting up reaction templates. A lot of the
torsions, ring breaks and comparison atoms will have to be set or at least
checked manually.


### 3. Merging structures

You're going to use `merge.py` to combine structures. Look at that script's help
and documentation for more information. `merge.py` uses information inside the
*.mae files to determine how to superposition structures and delete duplicate
atoms.

Here's an example:

```
python merger.py -g reactions/rh_hydrogenation_enamides.mae
-g substrates/*.mae -g ligands/*.mae
```

This will take the structures from `substrates/*.mae` and add them on top of
`reactions/rh_hydrogenation_enamides.mae`. Then it will take all
those combined structures and add all the structures from
`ligands/*.mae` on top of them. You can add as many structures as
you'd like this way.

Here's another example:

```
python merger.py -g base1.mae base2.mae -g next1.mae -g another1.mae
another2.mae -g evenmore1.mae
```

This would do the following:

Add group 1:
 * base1
   base2
Add group 2:
   * base1-next1
   * base2-next1
Add group 3:
     * base1-next1-another1.mae
     * base1-next1-another2.mae
     * base2-next1-another1.mae
     * base2-next1-another2.mae
Add group 4:
     * base1-next1-another1-evenmore1.mae
     * base1-next1-another2-evenmore1.mae
     * base2-next1-another1-evenmore1.mae
     * base2-next1-another2-evenmore1.mae

In order to make this work, the *.mae structures that are being added need
to have certain properties. You can use `s_cs_pattern` or `s_cs_pattern1` or
`s_cs_pattern2` as a few examples. Basically, any string containing "pattern"
will work. These are essentially SMILES-like strings that indicate a pattern
of atoms. Here's 2 examples:

* O=C-N-C=C
* P-[Rh]-P

It will search for these atoms in both structures, superposition those atoms
as best as Schrodinger can, and then merge the two structures. If properties
exist on the new atoms, those from the structure listed last by the `-g`
grouping, it will attempt to copy them over.

### 4. Setting up the conformational search files

All of the information inside the *.mae files allows for `setup_com_from_mae` to
automatically generate *.com files to do sampling on the merged structure.

```
python setup_com_from_mae.py mergedfile.mae newcom.com outputmae.mae
```

You can also setup the *.com files for many *.mae files simultaneously using
`setup_com_from_mae_many`.

```
python setup_com_from_mae_many.py somedir/*.mae

Say you had the files `maefilea.mae` and `maefileb.mae`, this would generate
`maefilea_cs.com` and `maefileb_cs.com`, which produce the output files
`maefilea_cs.mae` and `maefileb_cs.mae`.
```
