# Most of the description below comes from the original Schrodigner atom.typ
# file.

# This is a free-format file; fields may be separated by an arbitrary
#  number of blanks, but TABS MAY NOT BE USED. Blank lines are
#  gracefully ignored, as are leading blanks in a line.

# An unused, blank or unspecified field should be signified by an exclamation
#  point (!), which acts as a place-holder.

# Comments start with the character '#' and end with the end of the line.
#  A comment may appear after all valid entries on a line.

# Descriptions of each atom type are contained on a line preceeding it and
# beginning with the string "#&". If an atom type does not have a description
# then the most recent description will be used (so it's a good idea to have 
# them for all types). 

# Each line describes an atom type.  The entries on each line are as
#  follows.

# COLUMN 1:  Atom-type number.  This can be any number from 1 up to
#  a compiled-in maximum value, currently 300.  We would like to reserve
#  values up to 200 for expansion of MacroModel "standard" types and
#  301 through 400 for additional atom types;  therefore,
#  users wishing to add new types should restrict themselves to the range
#  201 through 300, inclusive.

#  This is where we are adding all of the Q2MM atom types.

# COLUMN 2: Atomic number. For a united atom, this is the atomic number of the 
#  underlying heavy atom.  Lone pairs get atomic number -1, dummy atoms -2.
#  The values -3 and 0 are reserved for use by MacroModel; user attempts
#  to specify these may or may not lead to desired effects.  I.e., please
#  avoid them.

# COLUMN 3: Atomic mass, in a.m.u.;  include mass of H's for united types.

# COLUMN 4: A unique two-character name for the type.  Care should be taken
#  not to conflict with either names of other types, or with names used for
#  aliases of atom equivalence classes in the force fields.

# COLUMN 5: The default color for MacroModel display.

# COLUMN 6: Van-der-Waals radius (used only in certain fall-back situations;
#  energetic calculations generally use values from .fld and .slv files).

# N.B.: Columns 7, 8, 9 and 10 are used in formal-charge delocalization.

# COLUMN 7: "T" if formal-charge delocalization is allowed over substituents, 
#  as in the central C in an allyl carbocation; otherwise, "F" or "!".

# COLUMN 8: "T" if, as a substituent atom, this atom type may participate
#  in delocalization of positive formal charges;  "F" or "!" otherwise.
#  Essentially, this column should get a "T" if, when initially single-bonded,
#  this atom type can become positively charged in a resonance structure;  
#  for example, Br in:
#   H-N+(-H)=C(-H)-Br <---> H-N(-H)-C(-H)=Br+

# COLUMN 9: Two-letter name of atom type to which this type should be
#  considered equivalent if they are in 1-3 bonded disposition.  The only
#  obvious case is =O ('O2')and O- ('OM'), which are equivalent if they are in, 
#  say, a phosphate or a carboxylate.

# COLUMN 10: Pauling electronegativity of this atom type.

# COLUMN 11: "T" if this atom type can serve as a wild-card type in the 
#  force-field for other types of the same atomic number;  examples: 
#  C0, N0, etc.  Usually the "simplest" of a group of types sharing the 
#  same atomic number.

# COLUMN 12: For any united atom, the two-letter name of the underlying non-
#  united type with the same atomic number and hybridization; for example, all
#  sp3 united-atom carbons should have "C3" in this column.

# COLUMN 13: For any united atom, the number of H's that distinguishes it
#  from the root type given in COLUMN 12.  For example, CA is an
#  sp3 united CH;  it should have "C3" in column 12 and "1" in column 13.

# COLUMN 14: PDB atom name for any atom type that is associated with a unique
#  pdb atom name.  Since a PDB atom name is a 4-char field, we need a way of
#  encoding blanks within the name.  A dot ('.') in the name means a blank.
#  Thus, calcium might be encoded as 'CA..', which adheres to the PDB
#  convention, and distinguishes it from an alpha-carbon, which is '.CA.'.
#  If there are no dots in the name, it will be considered to represent a
#  left-aligned field; that is, 'CA' is equivalent to 'CA..'.

#  For "standard" atom types, this field should be unspecified ('!'), since
#  for standard types there is not a one-to-one correspondence between PDB
#  atom names and MMOD atom names.  

# COLUMN 15: atom type which should be deemed equivalent for solvation
#  calculations, if the two types are 1-3 bonded. (4jul96: no longer used,
#  I think -- PSS.)

# COLUMN 16: Formal charge on this atom type.  Previously, this was
#  obtained from the force-field.

# COLUMN 17: The geometry of the atom (for atoms with valence > 1). Possible
#  valences are: 
#   NCA - not a central atom (valence = 1 or 0)
#   LIN - Linear
#   TRI - Trigonal
#   TET - Tetrahedral
#   TBY - Trigonal-bipyramidal
#   OCT - Octahedral

# COLUMN 18: The normal valence of this atom type. For united atom types then
#  the valence refers to the associated all atom type.

# COLUMN 19: The type of hydrogen which will be added in an HADD operation.

# Header lines follow:
#      1     2          3     4     5        6      7      8      9     10    11      12    13        14     15   16   17      18 19
# at_typ at_no      at_wt  name color  vdw_rad  deloc catdel 1-3eqv el_neg  wild UA_root  no_H  pdb_name 1-3slv fchg geom valence 

################################################################################
# Atom types defined for Q2MM

# Try to use these general types.
#&Palladium (Heck, Allylpalladium)
201      46  106.42         Pd    20     1.70      T      F      !  2.2       T       !     !         !      !   +1  NCA       2   !
#&Ru (Heck)
202      44  101.07         Ru    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !
#&Rh (Heck)
203      45  102.91         Rh    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !
#&Ir (Heck)
204      77  192.22         Ir    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !
#&Dummy atom (Ferrocene)
205      -2    0.0000       D1    10     1.00      F      F      !  0.0       T       !     !         !      !    0  NCA       1   !
#$Osmium (Heck)
206      76  190.23         Os    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !

# Atom types specific for Rh catalyzed hydrogenation of enamides.
# Need the +1 Rh, but the other atom types should be depreciated.
#&Rh catalyzed hydrogenation of enamides: Rh
250        !    102.90550    Rh     4     1.75      T      F      !   1.50     T       !     !         !      !   +1  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: H-moving hydrogen
251        1      1.00797    H7    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       2   !
#&Rh catalyzed hydrogenation of enamides: H-stationary hydrogen 
252        1      1.00797    H6    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: Phosphorus trans-O
253      15     30.97380    P2    15     1.80      T      F      !   2.10     F       !     !         !      !    0  TET       3   H1
#&Rh catalyzed hydrogenation of enamides: Phosphorus cis-O
254      15     30.97380    P1    15     1.80      T      F      !   2.10     F       !     !         !      !    0  TET       3   H1

################################################################################
