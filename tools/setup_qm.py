import argparse
import textwrap
import sys
import os

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, measure, rmsd

def get_sch_structs(filename, first_struct_only=True):
#    structures = {}
    structures = []
    with sch_struct.StructureReader(filename) as f:
        for struct in f:
            structures.append(struct)
#            ECP = False
#            for atom in struct.atom:
#                if atom.element in METALS:
#                    ECP = True
#            structures[struct] = ECP
            if first_struct_only:
                break
    return structures

def get_atoms_from_schrodinger_struct(structures_list):
    structures = {}
    for struct in structures_list:
        atom_list = []
        for atom in struct.atom:
            atom_list.append((atom.element, atom.x, atom.y, atom.z))
        if 'r_mmod_Potential_Energy-MM3*' in struct.property:
            energy = struct.property['r_mmod_Potential_Energy-MM3*']
            structures[struct] = [struct.formal_charge, atom_list, energy]
        else:
            structures[struct] = [struct.formal_charge, atom_list]
    structures_list = structures
    return structures_list



class GaussCom():
    def __init__(self, filename, atom_list, calculation_type,
        frozen_atoms=None, memory=8, procs=8, chk=None, frequency='',
        method='m06', basis='6-31+g*', ECP=False):
        self.filename = filename
        self.atom_list = atom_list
        self.calculation_type = calculation_type
        self.frozen_atoms = frozen_atoms
        self.memory = memory
        self.procs = procs
        self.chk = chk
        self.frequency = frequency
        self.method = method
        self.basis = basis
        self.ECP = ECP

    def write_com(self):
        with open(self.filename + '.com', 'w') as f:
            if self.chk:
                f.write('%chk={}\n'.format(self.chk))
            else:
                print("Checkpoint file is not give: continuing anyway")
            f.write('%mem={}GB\n'.format(self.memory))
            f.write('%nprocshared={}\n'.format(self.procs))
            # I'm not sure why but I guess I have to conver the textwrap to a
            # string inorder to write it.
            #f.write(str(textwrap.wrap(' '.join(['#',
            # I want to be able to wrap text, but I can't quite figure how to
            # write the wrapped text to a file
            f.write(' '.join(['#',
                'empiricaldispersion=gd3 int=ultrafine',
                self.basis,
                self.method,
                self.frequency,
                '\n\n'
                ]))
            f.write('SOME SORT OF STRING\n\n')
            for i,atom in enumerate(self.atom_list):
                f.write('   '.join((' {:2}'.format(atom[0]),
                                    '{:14.10f}'.format(atom[1]),
                                    '{:14.10f}'.format(atom[2]),
                                    '{:14.10f}'.format(atom[3]),
                                    '\n'
                                    )))
            f.write('\n')
            if self.frozen_atoms:
                print("do something here I don't know what yet")
            if self.ECP:
                print("do something here I don't know what yet")
       
         







def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)
    #Do I need try?
    if opts.calculationtype not in ['TS','GS','FZTS','FZGS']:
        raise Exception('Indicate a correction calculation type from: \
                        TS, GS, FZTS, FZGS.')
    for filename in opts.filename:
        structures = get_sch_structs(filename,first_struct_only=opts.allstructs)
        structures = get_atoms_from_schrodinger_struct(structures)
        if len(structures) == 1:
            for structure in structures:
                gaussian_file = GaussCom(filename=os.path.splitext(filename)[0],
                                    atom_list=structures[structure][1],
                                    calculation_type=opts.calculationtype)
                # Determine the basis set requested and if and ECP is needed.
                if opts.basisset:
                    if '/' in opts.basisset:
                        gaussian_file.basis = opts.basisset.split('/')
                        gaussian_file.ECP = True
                    else:
                        gaussian_file.basis = opts.basisset
                if opts.methodfunction:
                        gaussian_file.method = opts.methodfunction
                if opts.frequency:
                        gaussian_file.frequency = 'freq=noraman'
                if not opts.checkpoint:
                        gaussian_file.chk = gaussian_file.filename + '.chk'
                gaussian_file.write_com()                        
                                

def return_parser():
    parser = argparse.ArgumentParser(description="Automatically make a \
        gaussian com. This is inteded to be used with our virtual screening \
        methods either after templating all the component structures or \
        after conformational search. This can also be used to convert any \
        Schrodinger structures to guassian com files.")
    parser.add_argument('filename', nargs='+', type=str,
        help='A single or mutliple *.mae files (can also be *.maegz) that will\
            be converted into a Gaussian *.com file. This file can also \
            contain one or mutliple structures. The gaussian *.com file will \
            will share the same parent filename with the ".com" extension and \
            if multiple structures are being converted then the *.com files \
            will be named <parentfilename>_XXX.com where XXX is a number.')
    parser.add_argument('-ct', '--calculationtype', type=str, help='Must be \
            one of these arguments: TS, GS, FZTS, or FZGS. TS and FZTS will \
            optimize to a transition state where FZ indicates frozen coords. \
            GS and FZGS will optimize to a ground state. In the case of a TS \
            optimization, frequency calculation will be included by default.')
    parser.add_argument('-bs', '--basisset', type=str, help='Include a basis \
            set. If and ECP is needed seperate basis sets with "/". Example. \
            " -bs 6-31+g*/LANL2DZ". The ECP basis should always be last. The \
            default is 6-31+g*')  
    parser.add_argument('-mf', '--methodfunction', type=str, help='Include the\
            QM method you would like to use. The default will be M06.')
    parser.add_argument('-freq', '--frequency', action='store_true', help='If\
            included then a frequency calculation will be included after the \
            optimization.')
    parser.add_argument('-chk', '--checkpoint', action='store_false',
            help='If included then a checkpoint file will not be used during\
            the gaussian calculation.')
    parser.add_argument('-all', '--allstructs', action='store_false',
            help='Include this argument if you just want the minimum structure\
            to optimize at the QM level.')
    return parser

if __name__ == '__main__':
    main(sys.argv[1:])



