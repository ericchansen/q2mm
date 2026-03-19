#################################################################
# *** $RCSfile: atom.typ,v $
# *** $Revision: 1.53 $
# *** $Date: 2009/04/24 03:42:43 $
#################################################################

# atom.typ: information on individual atom types.
# version: 6.51

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
# them for all types... 

# Each line describes an atom type.  The entries on each line are as
#  follows.

# COLUMN 1:  Atom-type number.  This can be any number from 1 up to
#  a compiled-in maximum value, currently 300.  We would like to reserve
#  values up to 200 for expansion of MacroModel "standard" types and
#  301 through 400 for additional atom types;  therefore,
#  users wishing to add new types should restrict themselves to the range
#  201 through 300, inclusive.
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
#   this atom type can become positivelyh charged in a resonance structure;  
#   for example, Br in:
#	  H-N+(-H)=C(-H)-Br  <---> H-N(-H)-C(-H)=Br+
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
# For "standard" atom types, this field should be unspecified ('!'), since
#  for standard types there is not a one-to-one correspondence between PDB
#  atom names and MMOD atom names.  
# COLUMN 15: atom type which should be deemed equivalent for solvation
#  calculations, if the two types are 1-3 bonded.  (4jul96: no longer used,
#  I think -- PSS.)
# COLUMN 16: Formal charge on this atom type.  Previously, this was
#  obtained from the force-field.
# COLUMN 17: The geometry of the atom (for atoms with valence > 1). Possible
#  valences are: 
#	NCA - not a central atom (valence = 1 or 0)
#   LIN - Linear
#   TRI - Trigonal
#   TET - Tetrahedral
#   TBY - Trigonal-bipyramidal
#   OCT - Octahedral
# COLUMN 18: The normal valence of this atom type. For united atom types then
#   the valence refers to the associated all atom type.
# COLUMN 19: The type of hydrogen which will be added in an HADD operation.
#header lines follow:
#     1     2          3     4     5        6      7      8      9     10    11      12    13        14     15   16  17       18   19
#at_typ at_no      at_wt  name color  vdw_rad  deloc catdel 1-3eqv el_neg  wild UA_root  no_H  pdb_name 1-3slv fchg geom valence 

# Carbon, C:
#&Carbon - sp
1         6     12.01115    C1     2     1.78      T      F      !   2.50     F      C1     0         !      !    0  LIN       4   H1
#&Carbon - sp2
2         6     12.01115    C2     2     1.72      T      F      !   2.50     F      C2     0         !      !    0  TRI       4   H1
#&Carbon - sp3
3         6     12.01115    C3     2     1.70      F      F      !   2.50     F      C3     0         !      !    0  TET       4   H1
#&United atom CH -  sp3
4         6     13.01912    CA     2     1.80      F      F      !   2.50     F      C3     1         !      !    0  TET       4   H1
#&United atom CH2 -  sp3
5         6     14.02709    CB     2     1.90      F      F      !   2.50     F      C3     2         !      !    0  TET       4   H1
#&United atom CH3 -  sp3
6         6     15.03506    CC     2     2.00      F      F      !   2.50     F      C3     3         !      !    0  TET       4   H1
#&United atom CH -  sp2
7         6     13.01912    CD     2     1.80      T      F      !   2.50     F      C2     1         !      !    0  TRI       4   H1
#&United atom CH2 -  sp2
8         6     14.02709    CE     2     1.80      T      F      !   2.50     F      C2     2         !      !    0  TRI       4   H1
#&United atom CH -  sp
9         6     13.01912    CF     2     1.70      T      F      !   2.50     F      C1     1         !      !    0  LIN       4   H1
#&Carbanion
10        6     12.01115    CM     2     1.72      F      F      !   2.50     F      CM     !         !      !   -1  TRI       3   H5
#&Carbocation 
11        6     12.01115    CP     2     1.72      F      T      !   2.50     F      CP     !         !      !   +1  TRI       3   H4
#&Carbon free radical
12        6     12.01115    CR     2     1.72      F      F      !   2.50     F      CR     !         !      !    0  TRI       3   H1
#& 
13        6      0.00000    !      2     1.72      F      F      !   2.50     F       !     !         !      !    0  NCA       1   H1
#&Any Carbon
14        6     12.01115    C0     2     1.70      F      F      !   2.50     T      C3     !         !      !    0  TET       4   H1

# Oxygen, O:
#&Oxygen - double bond
15        8     15.99940    O2    70     1.50      T      T     OM   3.50     F       !     !         !      !    0  NCA       2   !
#&Oxygen - single bonds
16        8     15.99940    O3    70     1.52      F      F      !   3.50     F      O3     0         !      !    0  TET       2   H2
#&United atom OH
17        8     17.00737    OA    70     1.60      F      F      !   3.50     F      O3     1         !      !    0  TET       2   H2
#&O- (alkoxide, carboxylate)
18        8     15.99940    OM    70     1.70      F      F     O2   3.50     F      OM     !         !     O2   -1  LIN       1   H2
#&United atom H2O - Water
19        8     18.01534    OW    70     1.80      F      F      !   3.50     F      OW     !         !      !    0  TET       2   H2
#&Oxonium ion - sp2
20        8     15.99940    OP    70     1.52      F      F      !   3.50     F      OP     !         !      !    1  TRI       3   H4
#&Oxonium ion - sp3
21        8     15.99940    OQ    70     1.52      F      F      !   3.50     F      OQ     !         !      !    1  TET       3   H4
#& 
22        8      0.00000    !     70     1.52      F      F      !   3.50     F       !     !         !      !    0  NCA       1    !
#&Any Oxygen
23        8     15.99940    O0    70     1.60      F      F      !   3.50     T      O3     !         !      !    0  TET       2   H2

# Nitrogen, N:
#&Nitrogen - SP 
24        7     14.00670    N1     43     1.55      T      T      !   3.00     F      N1     0         !      !    0  NCA       3   H3
#&N - SP2
25        7     14.00670    N2     43     1.55      T      T      !   3.00     F      N2     0         !      !    0  TRI       3   H3
#&N - SP3
26        7     14.00670    N3     43     1.60      F      T      !   3.00     F      N3     0         !      !    0  TET       3   H3
#&United atom NH  - sp3
27        7     15.01467    NA     43     1.70      F      T      !   3.00     F      N3     1         !      !    0  TET       3   H3
#&United atom NH2  - sp3
28        7     16.02264    NB     43     1.75      F      T      !   3.00     F      N3     2         !      !    0  TET       3   H3
#&United atom NH  - sp2
29        7     15.01467    NC     43     1.70      T      T      !   3.00     F      N2     1         !      !    0  TRI       3   H3
#&United atom NH2  - sp2
30        7     16.02264    ND     43     1.75      T      T      !   3.00     F      N2     2         !      !    0  TRI       3   H3
#&N+ - SP2
31        7     14.00670    N4     43     1.55      T      F      !   3.00     F      N4     0         !      !   +1  TRI       4   H4
#&N+ - SP3
32        7     14.00670    N5     43     1.55      F      F      !   3.00     F      N5     0         !      !   +1  TET       4   H4
#&United atom NH+ -  sp3
33        7     15.01467    NE     43     1.60      F      F      !   3.00     F      N5     1         !      !   +1  TET       4   H4
#&United atom NH2+ - sp3
34        7     16.02264    NF     43     1.70      F      F      !   3.00     F      N5     2         !      !   +1  TET       4   H4
#&United atom NH3+ - sp3
35        7     17.03061    NG     43     1.80      F      F      !   3.00     F      N5     3         !      !   +1  TET       4   H4
#&United atom NH+ -  sp2
36        7     15.01467    NH     43     1.60      T      F      !   3.00     F      N4     1         !      !   +1  TRI       4   H4
#&United atom NH2+ - sp2
37        7     16.02264    NI     43     1.70      T      F      !   3.00     F      N4     2         !      !   +1  TRI       4   H4
#&N- sp3
38        7     14.00670    NM     43     1.55      F      F      !   3.00     F      NM     0         !      !   -1  TET       2   H5
#&N- sp2
39        7     14.00670    NP     43     1.55      F      F      !   3.00     F      NP     0         !      !   -1  NCA       2   H5
#&Any Nitrogen
40        7     14.00670    N0     43     1.55      F      F      !   3.00     T      N3     !         !      !    0  TET       3   H3

# Hydrogen, H:
#&H-Electroneutral (e.g. C,S) 
41        1      1.00797    H1    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&H-O(Neut)
42        1      1.00797    H2    21     1.00      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&H-N(Neut)
43        1      1.00797    H3    21     1.10      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&H-Cation
44        1      1.00797    H4    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&H-Anion
45        1      1.00797    H5    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#& 
46        1      1.00797    !     21        !      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#& 
47        1      1.00797    !     21        !      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&Any Hydrogen
48        1      1.00797    H0    21     1.20      F      F      !   2.00     T       !     !         !      !    0  NCA       1   !

# Sulfur, S:
#&Sulfur neutral
49       16     32.06400    S1    13     1.80      T      T      !   2.50     F      S1     0         !      !    0  TET       2   H1
#&United atom SH 
50       16     33.07197    SA    13     2.00      T      T      !   2.50     F      S1     1         !      !    0  TET       2   H1
#&Thiolate anion
51       16     32.06400    SM    13     1.80      F      F      !   2.50     F      SM     !         !      !   -1  LIN       1   H1
#&Any Sulfur
52       16     32.06400    S0    13     1.80      T      F      !   2.50     T      S1     !         !      !    0  TET       2   H1
# Phosphorus, P:
#&Phosphorus, trivalent
53       15     30.97380    P3    15     1.80      T      F      !   2.10     T      P3     !         !      !    0  TET       3   H1
# Boron: B
#&Boron, trigonal planar
54        5     11.00090    B2    10     1.75      F      F      !   0.00     F      B2     !         !      !    0  TRI       3   H1
#&Boron anion, tetrahedral
55        5     11.00090    B3    10     1.75      F      F      !   0.00     F      B3     !         !      !   -1  TET       4   H1

# Halogens:
#&Fluorine 
56        9     18.99840    F0    8     1.47      T      T      !   4.00     T       !     !         !      !    0  LIN       1   !
#&Chlorine
57       17     35.45300    Cl    9     1.75      T      T      !   3.00     T       !     !         !      !    0  LIN       1   !
#&Bromine
58       35     79.90900    Br    22     1.85      T      T      !   2.80     T       !     !         !      !    0  LIN       1   !
#&Iodine 
59       53    126.90440    I0    19     1.98      T      T      !   2.50     T       !     !         !      !    0  LIN       1   !

# Silicon: Si:
#&Silicon  
60       14     28.08600    Si    14     2.10      T      F      !   1.80     T      Si     !         !      !    0  TET       4   H1

# Dummy atom (for FEP): Du;  these are given atomic number -2:
#&Special dummy atom type (FEP)
61       -2     12.00000    Du    10     1.00      F      F      !   0.00     T       !     !         !      !    0  NCA       1   !

# Atom to be defined: Z0; perhaps no longer needed with increased atm. types:
#&Special Atom Type
62        !    100.00000    Z0     4     1.75      T      F      !   1.50     T       !     !         !      !    0  NCA       1   !

# Lone pair: Lp;  these get atomic number -1:
#&Lone Pair
63       -1     12.00000    Lp    14     0.50      F      F      !   3.50     T       !     !         !      !    0  NCA       1   !

# Any atom: 00;  used as a wildcard in the force-field:
#&Any Atom
64        !     12.00000    00     4     0.00      T      F      !   0.00     T       !     !         !      !    0  NCA       1   !

# el_neg are from Levine, P.Chem., p655, but
#  are wrong, since the values refer to the elements, not the ions.  But they'll probably
#  never be used in the program.  Cs and Ba el_neg's are guesses.:
# Radii for ions are ionic radii and are from the 1995 CRC handbook
# Alkali:
#&Lithium +
65        3      6.94100    Li     4     0.59      F      F      !   1.00     T       !     !      LI..      !   +1  NCA       1   !
#&Sodium +
66       11     22.99000    Na     4     1.02      F      F      !   0.90     T       !     !      NA..      !   +1  NCA       1   !
#&Potassium +
67       19     39.09800    K0     4     1.51      F      F      !   0.80     T       !     !      K...      !   +1  NCA       1   !
#&Rubidium +
68       37     85.47000    Rb     4     1.61      F      F      !   0.80     T       !     !      RB..      !   +1  NCA       1   !
#&Cesium +
69       55    132.91000    Cs     4     1.74      F      F      !   0.70     T       !     !      CS..      !   +1  NCA       1   !

# Alkaline earth:
#&Calcium 2+
70       20     40.08000    Ca     4     1.00      F      F      !   1.00     T       !     !      CA..      !   +2  NCA       1   !
#&Barium 2+
71       56    137.34000    Ba     4     1.42      F      F      !   0.80     T       !     !      BA..      !   +2  NCA       1   !
#&Magnesium 2+
72       12     24.31000    Mg     4     0.72      F      F      !    !       T       !     !      MG..      !   +2  NCA       1   !
# Transition metals - 
# Radii given are ionic radii from 1995 CRC handbook and are given for the
#  "most likely" coordination state
# No attempt is made to include "electronegativity" for these metals
#&Manganese 2+
73       25     54.94000    M2     4     0.67      F      F      !    !       T       !     !      MN..      !   +2  PBY       7   !
#&Manganese 3+
74       25     54.94000    M3     4     0.58      F      F      !    !       F       !     !         !      !   +3  PBY       7   !
#&Manganese 4+
75       25     54.94000    M4     4     0.39      F      F      !    !       F       !     !         !      !   +4  CUB       8   !
#&Manganese 5+
76       25     54.94000    M5     4     0.33      F      F      !    !       F       !     !         !      !   +5  CUB       8   !
#&Manganese 6+
77       25     54.94000    M6     4     0.26      F      F      !    !       F       !     !         !      !   +6  9CD       9   !
#&Manganese 7+
78       25     54.94000    M7     4     0.25      F      F      !    !       F       !     !         !      !   +7  9CD       9   !
#&Iron 2+
79       26     55.85000    f2     4     0.61      F      F      !    !       T       !     !         !      !   +2  OCT       6   !
#&Iron 3+
80       26     55.85000    f3     4     0.55      F      F      !    !       F       !     !      FE..      !   +3  PBY       7   !
#&Cobalt 2+
81       27     58.93000    o2     4     0.65      F      F      !    !       T       !     !         !      !   +2  OCT       6   !
#&Cobalt 3+
82       27     58.93000    o3     4     0.55      F      F      !    !       F       !     !      CO..      !   +3  OCT       6   !
#&Nickel 2+
83       28     58.71000    n2     4     0.69      F      F      !    !       T       !     !      NI..      !   +2  OCT       6   !
#&Nickel 3+
84       28     58.71000    n3     4     0.56      F      F      !    !       F       !     !         !      !   +3  OCT       6   !
#&Copper +
85       29     63.55000    c1     4     0.60      F      F      !    !       T       !     !         !      !   +1  OCT       6   !
#&Copper 2+
86       29     63.55000    c2     4     0.57      F      F      !    !       F       !     !      CU..      !   +2  OCT       6   !
#&Zinc 2+
87       30     65.37000    Zn     4     0.74      F      F      !    !       T       !     !      ZN..      !   +2  CUB       8   !
#&Molybdenum 3+
88       42     95.94000    m3     4     0.69      F      F      !    !       T       !     !         !      !   +3  CUB       8   !
#&Molybdenum 4+
89       42     95.94000    m4     4     0.65      F      F      !    !       F       !     !      MO..      !   +4  CUB       8   !
#&Molybdenum 5+
90       42     95.94000    m5     4     0.61      F      F      !    !       F       !     !         !      !   +5  9CD       9   !
#&Molybdenum 6+
91       42     95.94000    m6     4     0.59      F      F      !    !       F       !     !         !      !   +6  9CD       9   !
#&Strontium 2+
92       38     87.62000    Sr     4     1.12      F      F      !   0.95     T       !     !         !      !   +2  NCA       1   !
#&Lithium neutral
93        3      6.94100    L0     4     0.59      F      F      !   1.00     T       !     !         !      !    0  NCA       1   !
#&Magnesium neutral
94       12     24.31000    M0     4     0.72      F      F      !    !       T       !     !         !      !    0  NCA       1   !
#&Arsenic Tetrahedral
95       33     74.92160    As     4     1.85      T      F      !    2.18    T       !     !         !      !    0  TET       5   H1 
# extra atom types for main group elements
# SP is a positively charged sulfur
#&Sulfur cation
100      16     32.06400    SP    13     1.80      F      F      !   2.50     F      SP     !         !      !   +1  TET       3   H1
# S2 is an sp2 sulfur (as found in a thioketone)
#&Sulfur sp2 (thioketone)
101      16     32.06400    S2    13     1.80      T      F      !   2.50     F       !     !         !      !    0  NCA       2   !
# Cm is a negatively charged Cl - a chloride ion
#&Chloride ion
102      17     35.45300    Cm    13     1.75      T      T      !   3.00     F       !     !         !      !   -1  NCA       0   !
# BO is a Boron
#&Any Boron
103       5     11.00090    B0    10     1.75      F      F      !   0.00     T      B0     !         !      !    0  TRI       3   H1
# Fm is a negatively charged F - a fluoride ion
#&Fluoride ion
104       9     18.88400    Fm     8     1.47      T      T      !   4.00     F       !     !         !      !   -1  NCA       1   !
# Bm is a negatively charged Br - a bromide ion
#&Bromide ion
105      35     79.90900    Bm    22     1.85      T      T      !   2.80     F       !     !         !      !   -1  NCA       0   !
# Im is a negatively charged I - a iodide ion
#&Iodide ion
106      53    126.90440    Im    19     1.98      T      T      !   2.50     F       !     !         !      !   -1  NCA       1   !
# Pentavalent Phosphorus, P:
#&Phosphorus, pentavalent tetrahedral
107      15     30.97380    P5    15     1.80      T      F      !   2.10     F      P5     !         !      !    0  TET       5   H1
#&Any Phosphorus
108      15     30.97380    P0    15     1.80      T      F      !   2.10     T      P0     !         !      !    0  TET       3   H1
# More Sulfur, S:
#&Sulfur, tetravalent 
109      16     32.06400    S4    13     1.80      T      T      !   2.50     F      S4     0         !      !    0  TET       4   H1
#&Sulfur, hexavalent octohedral 
110      16     32.06400    S6    13     1.80      T      T      !   2.50     F      S6     0         !      !    0  OCT       6   H1
#&Phosphorus cation, tetravalent
111      15     30.97380    P4    15     1.80      T      F      !   2.10     F      P4     !         !      !   +1  TET       4   H1
#&Selenium neutral
112      34     78.96000    Se    13     1.80      T      T      !   2.55     T      Se     0         !      !    0  TET       2   H1
#&Sulfur, hexavalent tetrahedral
113      16     32.06400    ST    13     1.80      T      T      !   2.50     F      ST     0         !      !    0  TET       6   H1
#&Sulfur sulfide anion (-2)
114      16     32.06400    Sm    13     1.80      F      F      !   2.50     F      SM     !         !      !   -2  NCA       0   H1
#&Oxygen anion (-2)
115       8     15.99940    Om    70     1.70      F      F      !   3.50     F      Om     !         !      !   -2  NCA       0   !

################################################################################
# PI ligand dummy atom:
#&PI Dummy Atom
150       !       !         PI    15     1.50      F      F      !    !       F       !     !         !      !    0  NCA       0   !
# Generalized atom types follow:
# 
#&Isolated atom 
151       !       !         GA    20     1.50      F      F      !    !       F       !     !         !      !    0  NCA       0   !
#&Linear - single coordinate 
152       !       !         GB    20     1.50      F      F      !    !       F       !     !         !      !    0  NCA       1   !
#&Linear - two coordinate 
153       !       !         GC    20     1.50      F      F      !    !       F       !     !         !      !    0  LIN       2   !
#&Trigonal - two coordinate 
154       !       !         GD    20     1.50      F      F      !    !       F       !     !         !      !    0  TRI       2   !
#&Trigonal - three coordinate 
155       !       !         GE    20     1.50      F      F      !    !       F       !     !         !      !    0  TRI       3   !
#&Tetrahedral - three coordinate 
156       !       !         GF    20     1.50      F      F      !    !       F       !     !         !      !    0  TET       3   !
#&Tetrahedral - four coordinate 
157       !       !         GG    20     1.50      F      F      !    !       F       !     !         !      !    0  TET       4   !
#&Trigonal bipyramid - three coordinate 
158       !       !         GH    20     1.50      F      F      !    !       F       !     !         !      !    0  TBY       3   !
#&Trigonal bipyramid - four coordinate 
159       !       !         GI    20     1.50      F      F      !    !       F       !     !         !      !    0  TBY       4   !
#&Trigonal bipyramid - five coordinate 
160       !       !         GJ    20     1.50      F      F      !    !       F       !     !         !      !    0  TBY       5   !
#&Octahedral - four coordinate 
161       !       !         GK    20     1.50      F      F      !    !       F       !     !         !      !    0  OCT       4   !
#&Octahedral - five coordinate 
162       !       !         GL    20     1.50      F      F      !    !       F       !     !         !      !    0  OCT       5   !
#&Octahedral - six coordinate 
163       !       !         GM    20     1.50      F      F      !    !       F       !     !         !      !    0  OCT       6   !
#&Pentagonal bipyramid - seven coordinate 
164       !       !         GN    20     1.50      F      F      !    !       F       !     !         !      !    0  PBY       7   !
#&Twisted cube - eight coordinate 
165       !       !         GO    20     1.50      F      F      !    !       F       !     !         !      !    0  CUB       8   !
#&Nine coordinate 
166       !       !         GP    20     1.50      F      F      !    !       F       !     !         !      !    0  9CD       9   !
#&Ten coordinate 
167       !       !         GQ    20     1.50      F      F      !    !       F       !     !         !      !    0  ACD      10   !
#&Eleven coordinate 
168       !       !         GR    20     1.50      F      F      !    !       F       !     !         !      !    0  BCD      11   !
#&Icosahedron - twelve coordinate 
169       !       !         GS    20     1.50      F      F      !    !       F       !     !         !      !    0  CCD      12   !
#&Thirteen coordinate 
170       !       !         GT    20     1.50      F      F      !    !       F       !     !         !      !    0  DCD      13   !
#&Fourteen coordinate 
171       !       !         GU    20     1.50      F      F      !    !       F       !     !         !      !    0  ECD      14   !
#&Fifteen coordinate 
172       !       !         GV    20     1.50      F      F      !    !       F       !     !         !      !    0  FCD      15   !
#&Sixteen coordinate 
173       !       !         GW    20     1.50      F      F      !    !       F       !     !         !      !    0  GCD      16   !
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
#&Iron Ferrocene with better description
207      26     55.85000    Fe     4     1.75      T      F      !   1.83     T       !     !         !      !    0  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: Rh
250        !    102.90550    R2     4     1.75      T      F      !   1.50     T       !     !         !      !   +1  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: H-moving hydrogen
251        1      1.00797    H7    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       2   !
#&Rh catalyzed hydrogenation of enamides: H-stationary hydrogen
252        1      1.00797    H6    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: Phosphorus trans-O
253      15     30.97380    P2    15     1.80      T      F      !   2.10     F       !     !         !      !    0  TET       3   H1
#&Rh catalyzed hydrogenation of enamides: Phosphorus cis-O
254      15     30.97380    P1    15     1.80      T      F      !   2.10     F       !     !         !      !    0  TET       3   H1
################################################################################
# Generalized atom types for coarse grain sites
#     1     2          3     4     5        6      7      8      9     10    11      12    13        14     15   16  17       18   19
#
# Coarse grained solvent
#&Water: [H][O][H]
301       !     18.01534    X0    70     1.80      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&Hydroniuim ion: [H][O]([H])[H]
302       !     19.02331    X1    70     1.80      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&Hydroxyl ion: [O][H]
303       !     17.00737    X2    70     1.60      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&Place holder for future coarse grained site
304       !       !         X3     1     1.50      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&Place holder for future coarse grained site
305       !       !         X4     1     1.50      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Coarse grained alkanes
#&CH3- : [H][C]([H])([H])[*]
306       !     15.03506    X5    97     2.00      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH3-CH2 : [H][C]([H])([H])[C]([H])([H])[*]
307       !     29.06215    X6    97     2.46      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH3-CH2-CH2 : [H][C]([H])([H])[C]([H])([H])[C]([H])([H])[*]
308       !     43.08924    X7    97     2.79      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH3-CH2-CH2-CH2- : [H][C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[*]
309       !     57.11633    X8    97     3.06      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)CH- : [H][C]([C]([H])([H])[H])([C]([H])([H])[H])[*]
310       !     43.08924    X9    24     2.79      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)(CH3)C- : [H][C]([H])([H])[C]([C]([H])([H])[H])([C]([H])([H])[H])[*]
311       !     57.11633    XA    24     3.07      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)CH-CH2-CH2- : [H][C]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[C]([H])([H])[*]
312       !     71.14342    XB    24     3.29      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)(CH3)C-CH2- : [H][C]([H])([H])[C]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[*]
313       !     71.14342    XC    24     3.29      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH2- : [*][C]([H])([H])[C]([H])([H])[*]
314       !     28.05418    XD     2     2.39      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH2-CH2- : [*][C]([H])([H])[C]([H])([H])[C]([H])([H])[*]
315       !     42.08127    XE     2     2.74      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH2-CH2-CH2- : [*][C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[*]
316       !     56.10836    XF     2     3.02      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH(CH3)-CH2- : [*][C]([H])([C]([H])([H])[H])[C]([H])([H])[*]
317       !     42.08127    XG   103     2.75      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH(CH3)-CH2- : [*][C]([H])([H])[C]([H])([C]([H])([H])[H])[C]([H])([H])[*]
318       !     56.10836    XH   103     3.02      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH-CH2- : [*][C]([H])([H])[C]([H])([*])[C]([H])([H])[*]
319       !     41.07330    XI     2     2.69      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Coarse grained alkenes, alkynes
#&CH2=CH- : [H][C]([H])=[C]([H])[*]
326       !     27.04621    XP   104     2.33      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH=CH- : [*][C]([H])=[C]([H])[*]
327       !     26.03824    XQ    56     2.27      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH=CH-CH2- : [*][C]([H])=[C]([H])[C]([H])([H])[*]
328       !     40.06533    XR    56     2.65      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH=CH-CH2- : [*][C]([H])([H])[C]([H])=[C]([H])[C]([H])([H])[*]
329       !     54.09242    XS    56     2.94      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-C(CH3)=CH- : [*][C]([C]([H])([H])[H])=[C]([H])[*]
330       !     40.06533    XT    31     2.66      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Coarse grained aromatic compounds, carbon rings and other combinations
#&Benzene - C6H6 : [H]c1:c([H]):c([H]):c([H]):c([H]):c1([H])
336       !     78.11472    XZ   104     2.74      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&Phenyl group C6H5 : [*]c1:c([H]):c([H]):c([H]):c([H]):c1([H])
337       !     77.10675    Y0   104     2.69      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&>CH-CH< : [*][C]([H])([*])[C]([H])([*])[*]
338       !     26.03824    Y1     2     2.27      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&=CH-CH= : [*]=[C]([H])[C]([H])=[*]
339       !     26.03824    Y2    56     2.27      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH2-CH< : [*][C]([H])([H])[C]([H])([H])[C]([H])([*])[*]
340       !     41.07330    Y3     2     2.69      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&=CH-CH2-CH< : [*]=[C]([H])[C]([H])([H])[C]([H])([*])[*]
341       !     40.06533    Y4    56     2.65      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH2-CH= : [*][C]([H])([H])[C]([H])([H])[C]([H])=[*]
342       !     41.07330    Y5    56     2.69      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH2-C(CH3)< : [*][C]([H])([H])[C]([H])([H])[C]([C]([H])([H])[H])([*])[*]
343       !     55.10039    Y6   103     2.99      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH(CH3)-CH< : [*][C]([H])([H])[C]([H])([C]([H])([H])[H])[C]([H])([*])[*]
344       !     55.10039    Y7   103     2.98      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&=C-C(CH3)(CH2-)CH< : [*][C](=[*])[C]([C]([H])([H])[H])([C]([H])([H])[*])[C]([H])([*])[*]
345       !     66.10357    Y8   103     3.12      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Coarse grained hydroxyl group
#&HO- : [H][O][*]
351       !     17.00737    YE    70     1.60      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&HO-CH2- : [H][O][C]([H])([H])[*]
352       !     31.03446    YF    80     2.22      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&HO-CH2-CH2- : [H][O][C]([H])([H])[C]([H])([H])[*]
353       !     45.06155    YG    80     2.61      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH(OH)-CH2- : [*][C]([H])([H])[C]([H])([O][H])[C]([H])([H])[*]
354       !     58.08067    YH    87     2.87      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH(OH)-CH2-CH2- : [*][C]([H])([H])[C]([H])([O][H])[C]([H])([H])[C]([H])([H])[*]
355       !     72.10776    YI    87     3.12      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&glycerol - CH2(OH)-CH(OH)-CH2(OH) : [H][C]([O][H])([H])[C]([H])([O][H])[C]([H])([O][H])[H]
356       !     92.09541    YJ   101     3.17      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH2(OH)-CH(OH)-CH2- : [H][C]([O][H])([H])[C]([H])([O][H])[C]([H])([H])[*]
357       !     75.08804    YK   101     3.03      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH2(OH)-C(OH)-CH2(OH) : [H][C]([O][H])([H])[C]([O][H])([*])[C]([H])([O][H])[H]
358       !     91.08744    YL   101     3.14      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH2(OH)-CH(OH)-CH(OH)- : [H][C]([O][H])([H])[C]([H])([O][H])[C]([H])([O][H])[*]
359       !     91.08744    YM   101     3.13      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH2(OH)-CH(NH3)-CH(OH)- : [H][C]([O][H])([H])[C]([H])([N]([H])([H])[H])[C]([H])([O][H])[*]
360       !     91.11068    YN    22     3.19      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Coarse grained nitrogen group
#&NH3- : [H][N]([H])([H])[*]
366       !     17.03061    YT    33     1.80      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&NH3-CH2- : [H][N]([H])([H])[C]([H])([H])[*]
367       !     31.05770    YU    33     2.33      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#NH3-CH2-CH2- : [H][N]([H])([H])[C]([H])([H])[C]([H])([H])[*]
368       !     45.08479    YV    33     2.69      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-NH2-CH2- : [*][C]([H])([H])[N]([H])([H])[C]([H])([H])[*]
369       !     44.07682    YW    43     2.67      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-CH(NH3)-CH2- : [*][C]([H])([H])[C]([H])([N]([H])([H])[H])[C]([H])([H])[*]
370       !     57.09594    YX    43     2.94      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)NH2- : [H][N]([H])([C]([H])([H])[H])[*]
371       !     31.05770    YY   120     2.37      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)NH2-CH2-CH2- : [H][N]([H])([C]([H])([H])[H])[C]([H])([H])[C]([H])([H])[*]
372       !     59.11188    YZ   120     3.00      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)HN- : [H][N]([C]([H])([H])[H])([C]([H])([H])[H])[*]
373       !     45.08479    Z1   120     2.76      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)HN-CH2-CH2- : [H][N]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[C]([H])([H])[*]
374       !     73.13897    Z2   120     3.26      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)(CH3)N- : [H][C]([H])([H])[N]([C]([H])([H])[H])([C]([H])([H])[H])[*]
375       !     59.11188    Z3   120     3.04      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&(CH3)(CH3)(CH3)N-CH2-CH2- : [H][C]([H])([H])[N]([C]([H])([H])[H])([C]([H])([H])[H])[C]([H])([H])[C]([H])([H])[*]
376       !     87.16606    Z4   120     3.47      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Coarse grained acid groups
#&-O-P(=O)(O)-O- : [*][O][P](=[O])([O])[O][*]
381       !     94.97140    Z9    15     2.68      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-O-P(=O)(O)-O- : [*][C]([H])([H])[O][P](=[O])([O])[O][*]
382       !    108.99849    ZA    20     2.97      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-C(=O)O- : [*][C](=[O])[O][*]
383       !     44.00995    ZB   109     2.27      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&-CH2-C(=O)O- : [*][C]([H])([H])[C](=[O])[O][*]
384       !     58.03704    ZC   111     2.65      F      F      !    !       F       !     !         !      !    0  GCD      16   !
#&CH(C(=O)O)(NH3)-CH2- : [H][C]([C](=[O])[O])([N]([H])([H])[H])[C]([H])([H])[*]
385       !     88.08677    ZD    25     3.11      F      F      !    !       F       !     !         !      !    0  GCD      16   !

# Others
#&-NH-C(=O)- : [*][N]([H])[C](=[O])[*]
391       !     43.02522    ZJ   132     2.36      F      F      !    !       F       !     !         !      !    0  GCD      16   !
