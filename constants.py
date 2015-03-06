'''
Constants.
'''
import re

# default step sizes for parameters
steps = {'ae':      2.0,
         # 'af':      '0.1',
         'af':      0.2,
         'be':      0.1,
         # 'bf':      '0.1',
         'bf':      0.2,
         # 'df':     '0.01',
         'df':      0.2,
         'imp1':    0.2,
         'imp2':    0.2,
         'sb':      0.2,
         'q':       0.5}

# converts force constants from Jaguar frequency output (mdyn A**-1) to au
force_conversion = 15.56914
# converts eigenvalues of mass-weighted Hessian to cm**-1
eigenvalue_conversion = 53.0883777868
# converts Jaguar Hessian elements from au (Hartree bohr**-2) to
# kJ mol**-1 A**-2 (used by MacroModel)
hessian_conversion = 9375.829222
# convert Hartree to kJ/mol
hartree_to_kjmol = 2625.5

# matches any float
re_float = '[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# picks up smarts notation used by MM3* substructures
re_smiles = '[\w\-\=\(\)\.\+\[\]\*]+'
# characters used to split atoms in smarts notation
re_split_atoms = '[\s\-\(\)\=\.\[\]\*]+'
# picks up name of MM3* substructuers
re_sub = '[\w\s]+'

# picks up angle lines in a mmo file
re_angle = re.compile('\s+(\d+)\s+(\d+)\s+(\d+)\s+{0}\s+{0}\s+{0}\s+'
                      '({0})\s+{0}\s+{0}\s+\w+\s+\d+\s+({1})\s+(\d+)'.format(re_float, re_sub))
# picks up bond lines in a mmo file
re_bond = re.compile('\s+(\d+)\s+(\d+)\s+{0}\s+{0}\s+({0})\s+{0}\s+\w+'
                     '\s+\d+\s+({1})\s+(\d+)'.format(re_float, re_sub))


# string format used to write macromodel com files
format_macromodel = ' {0:4}{1:>8}{2:>7}{3:>7}{4:>7}{5:>11.4f}{6:>11.4f}{7:>11.4f}{8:>11.4f}\n'

# masses used for mass-weighting
masses = {
    'H':         1.007825032,
    'He':        4.002603250,
    'Li':        7.016004049,
    'Be':        9.012182135,
    'B':        11.009305466,
    'C':        12.000000000,
    'N':        14.003074005,
    'O':        15.994914622,
    'F':        18.998403205,
    'Ne':       19.992440176,
    'Na':       22.989769675,
    'Mg':       23.985041898,
    'Al':       26.981538441,
    'Si':       27.976926533,
    'P':        30.973761512,
    'S':        31.972070690,
    'Cl':       34.968852707,
    'Ar':       39.962383123,
    'K':        38.963706861,
    'Ca':       39.962591155,
    'Sc':       44.955910243,
    'Ti':       47.947947053,
    'V':        50.943963675,
    'Cr':       51.940511904,
    'Mn':       54.938049636,
    'Fe':       55.934942133,
    'Co':       58.933200194,
    'Ni':       57.935347922,
    'Cu':       62.929601079,
    'Zn':       63.929146578,
    'Ga':       68.925580912,
    'Ge':       73.921178213,
    'As':       74.921596417,
    'Se':       79.916521828,
    'Br':       78.918337647,
    'Kr':       83.911506627,
    'Rb':       84.911789341,
    'Sr':       87.905614339,
    'Y':        88.905847902,
    'Zr':       89.904703679,
    'Nb':       92.906377543,
    'Mo':       97.905407846,
    'Tc':       97.907215692,
    'Ru':      101.904349503,
    'Rh':      102.905504182,
    'Pd':      105.903483087,
    'Ag':      106.905093020,
    'Cd':      113.903358121,
    'In':      114.903878328,
    'Sn':      119.902196571,
    'Sb':      120.903818044,
    'Te':      129.906222753,
    'I':       126.904468420,
    'Xe':      131.904154457,
    'Cs':      132.905446870,
    'Ba':      137.905241273,
    'La':      138.906348160,
    'Ce':      139.905434035,
    'Pr':      140.907647726,
    'Nd':      141.907718643,
    'Pm':      144.912743879,
    'Sm':      151.919728244,
    'Eu':      152.921226219,
    'Gd':      157.924100533,
    'Tb':      158.925343135,
    'Dy':      163.929171165,
    'Ho':      164.930319169,
    'Er':      167.932367781,
    'Tm':      168.934211117,
    'Yb':      173.938858101,
    'Lu':      174.940767904,
    'Hf':      179.946548760,
    'Ta':      180.947996346,
    'W':       183.950932553,
    'Re':      186.955750787,
    'Os':      191.961479047,
    'Ir':      192.962923700,
    'Pt':      194.964774449,
    'Au':      196.966551609,
    'Hg':      201.970625604,
    'Tl':      204.974412270,
    'Pb':      207.976635850,
    'Bi':      208.980383241,
    'Po':      208.982415788,
    'At':      209.987131308,
    'Rn':      222.017570472,
    'Fr':      223.019730712,
    'Ra':      226.025402555,
    'Ac':      227.027746979,
    'Th':      232.038050360,
    'Pa':      231.035878898,
    'U':       238.050782583,
    'Np':      237.048167253,
    'Pu':      244.064197650,
    'Am':      243.061372686,
    'Cm':      247.070346811,
    'Bk':      247.070298533,
    'Cf':      251.079580056,
    'Es':      252.082972247,
    'Fm':      257.095098635,
    'Md':      258.098425321,
    'No':      259.101024000,
    'Lr':      262.109692000
    }

# in progress--working on grouped parameters
groups = {'bf': {'c': ' -d test -meig X001.mae,X001.01.out',
                 'r': ' -d test -jeige X001.02.in,X001.01.out'},
          'eb': {'c': ' -d test -mb X001.mae',
                 'r': ' -d test -jb X001.mae'}}
