Active site of a LPMO protein
===============

This example is taken from a research project on the LPMP protein. The active site coordinates are taken from the Protein databank file with ID [5ACG](https://www.rcsb.org/structure/5ACG)

## To calculate bond angle force field parameters:

For bonds...

```
seminario_ff -f model.fchk -s model.gro -b :1@N-:3@CU :1@ND1-:3@CU :2@NE2-:3@CU
```

...and for angles

```
seminario_ff -f model.fchk -s model.gro -a :1@CG-:1@ND1-:3@CU :1@CE1-:1@ND1-:3@CU
```

The atoms are specified as AMBER mask, see documentation for the [parmed package](http://parmed.github.io/ParmEd/) and delimited by dashes. Any number of bonds and angles can be specified.
