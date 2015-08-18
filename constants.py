'''
Constants and fairly basic variables used throughout Q2MM.
'''
import re

# Settings loaded using logging.config.
LOG_SETTINGS = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'bare': {'format': '%(message)s'},
        'basic': {'format': '%(name)s - %(message)s'},
        'simple': {'format': '%(asctime)s:%(name)s:%(levelname)s - %(message)s'}
        },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler', 'formatter': 'bare',
            'level': 'NOTSET'},
            # 'class': 'logging.StreamHandler', 'formatter': 'basic',
            # 'level': 'NOTSET'},
        'root_file_handler': {
            'class': 'logging.FileHandler', 'filename': 'root.log',
            'formatter': 'bare', 'level': 'NOTSET'}
            # 'class': 'logging.FileHandler', 'filename': 'root.log',
            # 'formatter': 'basic', 'level': 'NOTSET'}
        },
    'loggers': {'__main__': {'level': 'NOTSET', 'propagate': True},
                # 'calculate': {'level': 'NOTSET', 'propagate': True},
                # 'calculate': {'level': 20, 'propagate': True},
                'compare': {'level': 'NOTSET', 'propagate': True},
                # 'compare': {'level': 20, 'propagate': True},
                'datatypes': {'level': 'NOTSET',' propagate': True},
                # 'datatypes': {'level': 20,' propagate': True},
                'filetypes': {'level': 'NOTSET', 'propagate': True},
                # 'filetypes': {'level': 20, 'propagate': True},
                'gradient': {'level': 'NOTSET', 'propagate': True},
                'loop': {'level': 'NOTSET', 'propagate': True},
                # 'loop': {'level': 20, 'propagate': True},
                'opt': {'level': 'NOTSET', 'propagate': True},
                'simplex': {'level': 'NOTSET', 'propagate': True}
                },
    'root': {
        'level': 'NOTSET',
        'propagate': True,
        'handlers': ['console', 'root_file_handler']}
    }
# These are the initial step sizes used in numerical differentiation for all
# the different parameter types. Floats/integers or strings are accepted.
#
# When a float/integer step size is provided, the new, stepped parameter value
# is determined by simply adding or subtracting the step size for incrementing
# or decrementing, respectively.
#   x_new = x +/- step
#
# Instead, if a string is provided (by placing the number in single or double
# quotes), the new parameter value will be decremented or incremented by a
# percentage of its current value.
#   x_new = x +/- (x * step)
STEPS = {'ae':      2.0,
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
         'q':       0.5
         }
# Weights used in parameterization.
# Please figure out all units.
WEIGHTS = {'Angle':          5.0,
           'charge':         1.0,
           'Bond':         100.0,
           'eig_i':          1.0, # Weight of 1st eigenvalue.
           'eig_d':          1.0, # Weight of other eigenvalues.
           'eig_o':          0.5, # Weight of off diagonals in eigenmatrix.
           'energy_1':     100.0,
           'energy_2':       3.0,
           'energy_opt':   125.0,
           'Torsion':        5.0
           }
# String used to make a table in sqlite3 database.
STR_INIT_SQLITE3 = """DROP TABLE IF EXISTS data;
CREATE TABLE data (id INTEGER PRIMARY KEY AUTOINCREMENT, val REAL, wht REAL,
com TEXT, typ TEXT, src_1 TEXT, src_2 TEXT, idx_1 INTEGER, idx_2 INTEGER,
atm_1 INTEGER, atm_2 INTEGER, atm_3 INTEGER, atm_4 INTEGER)"""
# String for adding values to database.
STR_SQLITE3 = ('INSERT INTO data VALUES (:id, :val, :wht, :com, :typ, :src_1, '
               ':src_2, :idx_1, :idx_2, :atm_1, :atm_2, :atm_3, :atm_4)')
# These are the columns used in the sqlite3 database.
DATA_DEFAULTS = {'id': None,
                 'val': None, 'wht': None,
                 'com': None, 'typ': None,
                 'src_1': None, 'src_2': None,
                 'idx_1': None, 'idx_2': None,
                 'atm_1': None, 'atm_2': None, 'atm_3': None, 'atm_4': None}
def set_data_defaults(datum):
    '''
    Add keys and default values (None) to a data dictionary if they're missing.
    '''
    return {k: datum.get(k, DATA_DEFAULTS[k]) for k in DATA_DEFAULTS}
# Force constants from Jaguar frequency (mdyn A**-1) to au.
FORCE_CONVERSION = 15.56914
# Eigenvalues of mass-weighted Hessian to cm**-1.
EIGENVALUE_CONVERSION = 53.0883777868
# Hessian elements in au (Hartree Bohr**-2) from Jaguar to
# kJ mol**-1 A**-2 (used by MacroModel).
HESSIAN_CONVERSION = 9375.829222
# Hartree to kJ mol**-1.
HARTREE_TO_KJMOL = 2625.5
# Match any float in a string.
RE_FLOAT = '[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
# Match SMARTS notation used by MM3* substructures.
# More from import_ff could be added here.
RE_SMILES = '[\w\-\=\(\)\.\+\[\]\*]+'
# Possible symbols used to split atoms in SMARTS notation.
RE_SPLIT_ATOMS = '[\s\-\(\)\=\.\[\]\*]+'
# Name of MM3* substructures.
RE_SUB = '[\w\s-]+'
# Match bonds in lines of a .mmo file.
RE_BOND = re.compile('\s+(\d+)\s+(\d+)\s+{0}\s+{0}\s+({0})\s+{0}\s+\w+'
                     '\s+\d+\s+({1})\s+(\d+)'.format(RE_FLOAT, RE_SUB))
# Match angles in lines of a .mmo file.
RE_ANGLE = re.compile('\s+(\d+)\s+(\d+)\s+(\d+)\s+{0}\s+{0}\s+{0}\s+'
                      '({0})\s+{0}\s+{0}\s+\w+\s+\d+\s+({1})\s+(\d+)'.format(
        RE_FLOAT, RE_SUB))
# Match torsions in lines of a .mmo file.
RE_TORSION = re.compile('\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+{0}\s+{0}\s+{0}\s+'
                        '({0})\s+{0}\s+\w+\s+\d+({1})\s+(\d+)'.format(
        RE_FLOAT, RE_SUB))
# String format used to write MacroModel .com files.
FORMAT_MACROMODEL = ' {0:4}{1:>8}{2:>7}{3:>7}{4:>7}{5:>11.4f}{6:>11.4f}{7:>11.4f}{8:>11.4f}\n'
# Masses used for mass-weighting
MASSES = {
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

