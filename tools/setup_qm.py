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
        dummy = None
        for atom in struct.atom:
            if atom.atomic_number < 0:
                dummy = atom
            atom_list.append((atom.element, atom.x, atom.y, atom.z))
        if dummy:
            struct.deleteAtoms((dummy.index),renumber_map=False)
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
        method='m06', basis='6-31+g*', ECP=False, opt='', mae_struct=None):
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
        self.opt = opt
        self.mae_struct = mae_struct

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
            self.determine_method()
            route_section = ['#','empiricaldisperion=gd3','int=ultrafine',
                                 self.basis,self.method]
            if self.frequency:
                route_section.append(self.frequency)
            if self.opt:
                route_section.append(self.opt)
            route_section.append('\n\n')
            f.write(' '.join(route_section))
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
                for i,line in enumerate(self.frozen_atoms):
                    if i+1 == len(self.frozen_atoms):
                        f.write(line + '\n\n')
                    else:
                        f.write(line + '\n')
            if self.ECP:
                print("do something here I don't know what yet")
            if self.frozen_atoms:
                f.write('--link1-- \n')
                if self.chk:
                    f.write('%chk={}\n'.format(self.chk))
                f.write('%mem={}GB\n'.format(self.memory))
                f.write('%nprocshared={}\n'.format(self.procs))
                route_section = ['#','geom=allcheck','empiricaldisperion=gd',
                                 'int=ultrafine','chkbasis']
                if self.frequency:
                    route_section.append(self.frequency)
                if self.opt:
                    route_section.append(self.opt[:-1] + ',nofreeze)')
                route_section.append('\n\n\n')
                f.write(' '.join(route_section))
       
    def determine_method(self):
        if self.frozen_atoms:
            self.opt += ' geom=modredundant'
        if 'TS' in self.calculation_type: 
            if not self.frequency:
                self.frequency = 'freq=noraman'
            self.opt = 'opt=(calcfc,ts,noeigentest,maxcycle=50)'
        if 'GS' in self.calculation_type: 
            self.opt = 'opt=(calcfc,maxcycle=50)'

    def get_dictionary_of_frozen_coords(self,patterns):
        patterns_and_type = {}
        seperated_patterns = patterns.split(';')
        for pattern in seperated_patterns:
            key_value = pattern.split(',')
            patterns_and_type[key_value[0]] = key_value[1]
        return patterns_and_type
            
    def get_frozen_coords(self, dict_of_patterns, from_command_line=False):
        frozen_coord_lines = []
        if from_command_line:
            dict_of_patterns = self.get_dictionary_of_frozen_coords(
                                                              dict_of_patterns)
        for pattern in dict_of_patterns:
            atom_indicies = analyze.evaluate_substructure(self.mae_struct,
                                                    pattern,
                                                    first_match_only=False)[0]
            str_indicies = []
            for index in atom_indicies:
                str_indicies.append(str(index))
                
            frozen_coord_lines.append('{} {} F'.format(
                                        dict_of_patterns[pattern],
                                        ' '.join(str_indicies)))
        return frozen_coord_lines



def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)
    # I think it is better to have the single point be default if no option
    # is included. So the following code Should not be needed.
    #Do I need try?
    #if opts.calculationtype not in ['SP','TS','GS','FZTS','FZGS']:
     #   raise Exception('Indicate a correction calculation type from: \
     #                   TS, GS, FZTS, FZGS.')
    for filename in opts.filename:
        structures = get_sch_structs(filename,first_struct_only=opts.allstructs)
        structures = get_atoms_from_schrodinger_struct(structures)
        if len(structures) == 1:
            for structure in structures:
                gaussian_file = GaussCom(filename=os.path.splitext(filename)[0],
                                    atom_list=structures[structure][1],
                                    calculation_type=opts.calculationtype,
                                    mae_struct=structure)
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
                if opts.frozenatoms:
                    gaussian_file.frozen_atoms=gaussian_file.get_frozen_coords(
                                                        opts.frozenatoms,
                                                        from_command_line=True)

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
            one of these arguments: SP, TS, GS, FZTS, or FZGS. TS and FZTS will\
            optimize to a transition state where FZ indicates frozen coords. \
            GS and FZGS will optimize to a ground state. In the case of a TS \
            optimization, frequency calculation will be included by default. SP\
            will just perform a single point energy calculation.')
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
    parser.add_argument('-fa', '--frozenatoms', type=str, help='Include the \
            the atoms you wish to freeze using the Schrodinger substructure \
            language. Also include the type of coordinate (b, a, t). Use ";" \
            to seperate multiple coordinates. \
            Example: -fa " C2.C2,b;PD-C2.C2,a"')
    return parser

if __name__ == '__main__':
    main(sys.argv[1:])



