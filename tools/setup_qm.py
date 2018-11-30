import argparse
import textwrap
import sys
import os

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, measure, rmsd

class GaussCom():
    def __init__(self, filename, atom_list, calculation_type, charge,
        frozen_atoms=None, memory=8, procs=8, chk=None, frequency='',
        method='m06', basis='6-31+g*', ECP=False, opt='', mae_struct=None):
        self.filename = filename
        self.atom_list = atom_list
        self.calculation_type = calculation_type
        self.charge = charge
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

    def write_com(self,directory='./'):
        """
        This is the heart of the code where all of the options are used to 
        write the Gaussian command file.
        """
        with open(directory + self.filename + '.com', 'w') as f:
            if self.chk:
                f.write('%chk={}\n'.format(self.chk))
            f.write('%mem={}GB\n'.format(self.memory))
            f.write('%nprocshared={}\n'.format(self.procs))
            # I'm not sure why but I guess I have to conver the textwrap to a
            # string inorder to write it.
            #f.write(str(textwrap.wrap(' '.join(['#',
            # I want to be able to wrap text, but I can't quite figure how to
            # write the wrapped text to a file
            self.determine_method()
            # If we are using an ECP then we can't include the basis in the 
            # route section, and we have to use 'genecp'.
            if self.ECP:
                route_section = ['#','empiricaldispersion=gd3','int=ultrafine',
                                 'genecp',self.method]
            else:
                route_section = ['#','empiricaldispersion=gd3','int=ultrafine',
                                 self.basis,self.method]
            if self.frequency:
                route_section.append(self.frequency)
            # Specifies if we are doing a single point or some sort of geom
            # optimization.
            if self.opt:
                route_section.append(self.opt)
            route_section.append('\n\n')
            f.write(' '.join(route_section))
            f.write('SOME SORT OF STRING\n\n')
            f.write('{} 1 \n'.format(self.charge))
            # I'm not sure if maestro files give coordinates to 10 decimals, but
            # I have included it here because this class should be independent
            # of the source of the structure.
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
                # I'm not quite sure what the 0 does at the end of the element
                # lines. I don't think we will every have a nonzero value.
                f.write(' '.join(self.ECP[1:])+' 0\n')
                f.write(self.ECP[0] + '\n')
                f.write('****\n')
                nonECP = self.get_nonECP_atoms()
                f.write(' '.join(nonECP)+' 0\n')
                f.write(self.basis + '\n')
                f.write('****\n\n')
                f.write(' '.join(self.ECP[1:])+' 0\n')
                f.write(self.ECP[0] + '\n\n')
                 
            # If we have frozen coordinates then we probably only have them to
            # keep the geometery close to our starting structure. We then want
            # to do a second optimization but now without any forzen coords.
            if self.frozen_atoms:
                f.write('--link1-- \n')
                if self.chk:
                    f.write('%chk={}\n'.format(self.chk))
                f.write('%mem={}GB\n'.format(self.memory))
                f.write('%nprocshared={}\n'.format(self.procs))
                route_section = ['#','geom=allcheck','empiricaldispersion=gd3',
                                 'int=ultrafine','chkbasis',self.method]
                if self.frequency:
                    route_section.append(self.frequency)
                # This option is intended to optimize a structure to a GS but
                # with frozen coordinates to later optimize to a TS with no
                # frozen coordinates.
       # I think it is best to just do a frequency calcluations after a frozen
       # coordinate optimization, this way the user can check the vibrations
                if self.calculation_type == 'FZTS':
       #             route_section.append(
       #                     ' opt=(calcfc,ts,noeigentest,maxcycle=50,nofreeze)')
                    route_section.append(' freq=noraman')
       #         elif self.opt:
       #             route_section.append(self.opt[:-1] + ',nofreeze)')
                route_section.append('\n\n\n')
                f.write(' '.join(route_section))
    
    def get_nonECP_atoms(self):
        """
        The user should just have to specify what elements should be used for
        the ECP, and then this function will determine the rest of the elements
        for the other basis.
        """
        # Not sure if there is an instance where we want all of the atoms so 
        # I'm including it anyway if it is needed.
        atoms = []
        nonECPatoms =[]
        for atom in self.mae_struct.atom:
            if atom.element not in atoms:
                atoms.append(atom.element)
                if atom.element not in self.ECP:
                    nonECPatoms.append(atom.element)
        return nonECPatoms

    def determine_method(self):
        """
        Simple way to determine the type of calculation (SP, TS, GS, etc.)
        """
        # It might be better to collect all self.opt arguments as a list and 
        # then join() them to have the command file look prettier.
        if self.frozen_atoms:
            self.opt += ' geom=modredundant'
        if self.calculation_type == 'FZTS':
            self.opt += ' opt=(calcfc,maxcycle=500)'
        if self.calculation_type == 'TS':
            # Do I need this part? Will self.frequency ever be a value other 
            # than None or freq=noraman?
            if not self.frequency:
                self.frequency = 'freq=noraman'
            # The 50 cycles can probably be an option that is allowed to change,
            # but it seems fine for now.  
            self.opt += ' opt=(calcfc,ts,noeigentest,maxcycle=50)'
        if self.calculation_type == 'GS':
            self.opt += ' opt=(calcfc,maxcycle=50)'

    def get_dictionary_of_frozen_coords(self,patterns):
        """
        Converts the argparse argument from the command line to a dictionary
        for get_frozen_coords().
        
        Arguments
        ---------
        patterns : the string that is from the command line. This should be a
                   nonspace string where each coordinate that is intended to 
                   be forzen is sperated by ';' and the key and value are split
                   by ','.
        
        Returns
        -------
        patterns_and_type : a dictionary with pattern as the key and coordinate
                            type as the value.
        """
        patterns_and_type = {}
        seperated_patterns = patterns.split(';')
        for pattern in seperated_patterns:
            key_value = pattern.split(',')
            patterns_and_type[key_value[0]] = key_value[1]
        return patterns_and_type
            
    def get_frozen_coords(self, dict_of_patterns, from_command_line=False):
        """
        I'm still unsure if I want this within the class or outside. Either way
        This sets up the lines for frozen coordinates that will be written in 
        the command file.

        Arguments
        ---------
        dict_of_patterns: dictionary of patterns to match with schrodingers
                          psuedo SMILES language. {key=SMILE like pattern:
                          value=coordinate type such as bond or angle}        

        Returns
        -------
        frozen_coord_lines: A list of strings that will each be written on a 
                            seperate line of the Gaussian Command file.
        """
        frozen_coord_lines = []
        # This is the part I am not sure if I should include in the class 
        # instance or have it within main() since it has more to do with the
        # command line.
        if from_command_line:
            dict_of_patterns = self.get_dictionary_of_frozen_coords(
                                                              dict_of_patterns)
        for pattern in dict_of_patterns:
            matches = analyze.evaluate_substructure(self.mae_struct,
                                                    pattern,
                                                    first_match_only=False)
            for atom_indicies in matches:
                str_indicies = []
                for index in atom_indicies:
                    str_indicies.append(str(index))
                frozen_coord_lines.append('{} {} F'.format(
                                        dict_of_patterns[pattern],
                                        ' '.join(str_indicies)))
        return frozen_coord_lines

def get_sch_structs(filename, first_struct_only=True):
    """
    Arguments
    ---------
    filename : Maestro file. Supports both *mae and *maegz.
    first_struct_only : Gather only the first sturcture (True) or gather them
                        all (False). This should only be True when the user
                        knows the the first structure is the one they want. 
                        This is important, because sometimes the lowest energy
                        structure is not the first structure.
    
    Returns
    -------
    structers : A LIST of structure class instances from maestro.
    """
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
    """
    This gets the atom coordinates in a format for the GaussCom class. I have
    included energies of the structures (energies from MM calculations), but I
    have not included a means to use this information when making structures.


    Arguments
    ---------
    structures_list : A list of structure class instances from schrodinger.

    Returns
    -------
    structures_list : returns a dictionary with the following key:value combo:
            {Key=Structure class instance: Value=[charge,atom listings, energy]}    
    """
    structures = {}
    for struct in structures_list:
        atom_list = []
        dummy = []
        for atom in struct.atom:
            atom._setAtomType(atom.atom_type)
            if atom.atomic_number < 0:
                dummy.append(atom.index)
            atom_list.append((atom.element, atom.x, atom.y, atom.z))
        if dummy:
            struct.deleteAtoms(dummy,renumber_map=False)
        if 'r_mmod_Potential_Energy-MM3*' in struct.property:
            energy = struct.property['r_mmod_Potential_Energy-MM3*']
            structures[struct] = [struct.formal_charge, atom_list, energy]
        else:
            structures[struct] = [struct.formal_charge, atom_list]        
    structures_list = structures
    return structures_list


def main(args):
    """
    Uses the command line arguments to create a GaussCom instance in order to
    write the intended command file. 
    """
    parser = return_parser()
    opts = parser.parse_args(args)
    # I think it is better to have the single point be default if no option
    # is included. So the following code Should not be needed.
    #Do I need try?
    #if opts.calculationtype not in ['SP','TS','GS','FZTS']:
     #   raise Exception('Indicate a correction calculation type from: \
     #                   TS, GS, FZTS.')
    if opts.calculationtype == 'FZTS' and not opts.frozenatoms:
        raise Exception('The FZTS argument can only be used when frozen atoms' + 
                         ' are indicated.')
    for filename in opts.filename:
        structures = get_sch_structs(filename,first_struct_only=opts.allstructs)
        structures = get_atoms_from_schrodinger_struct(structures)

        for i,structure in enumerate(structures):
            print(' WRITING COM FILE FOR STRUCTURE {}'.format(i+1))
            if len(structures) == 1:
                gaussian_file = GaussCom(filename=os.path.splitext(filename)[0],
                                atom_list=structures[structure][1],
                                calculation_type=opts.calculationtype,
                                charge=structures[structure][0],
                                mae_struct=structure)
            else:
                file_iterator_name = os.path.splitext(filename)[0] + \
                    '_{0:03d}'.format(i+1)
                gaussian_file = GaussCom(filename=file_iterator_name,
                                atom_list=structures[structure][1],
                                calculation_type=opts.calculationtype,
                                charge=structures[structure][0],
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
            if opts.basisset:
                # I am unsure about the different scenarios that will be 
                # encountered when it comes to basis sets and ECPs. So this
                # is written to work with what we have done in the past 
                # with Q2MM.
                sets = opts.basisset.split('/')
                if len(sets) > 1:
                    ECP_string = sets[1].split(',')
                    ECP = [ECP_string[0]]
                    for element in ECP_string[1:]:
                        ECP.append(element)
                    gaussian_file.ECP = ECP
                gaussian_file.basis = sets[0]
            if opts.directory:
                gaussian_file.write_com(directory=opts.directory)
            else:
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
    parser.add_argument('-ct', '--calculationtype', type=str, default='SP',
            help='Must be one of these arguments: SP, TS, GS, or FZTS. \
            TS will optimize to a transition state. FZTS will optimize to a GS \
            with the frozen coordinates and then optimize the structure to a TS\
            without any frozen coords. GS will optimize to a ground state. In \
            the case of a TS optimization, frequency calculation will be \
            included by default. SP will just perform a single point energy \
            calculation. The default calculation if -ct is not present will be\
            a SP.')
    parser.add_argument('-bs', '--basisset', type=str, help='Include a basis \
            set. If an ECP is needed seperate basis sets with "/" and include \
            elements that are wanted for ECP with a ",". Example. \
             -bs "6-31+g*/LANL2DZ,Pd,Fe". The ECP basis should always be last. \
            The default is 6-31+g* without any basis.')  
    parser.add_argument('-mf', '--methodfunction', type=str, help='Include the\
            QM method you would like to use. The default will be M06.')
    parser.add_argument('-freq', '--frequency', action='store_true', help='If\
            included then a frequency calculation will be included after the \
            optimization.')
    parser.add_argument('-chk', '--checkpoint', action='store_false',
            help='If included then a checkpoint file will be used during\
            the gaussian calculation.')
    parser.add_argument('-all', '--allstructs', action='store_false',
            help='Include this argument if you just want all the structures\
            of a maestro file to optimize at the QM level.')
    parser.add_argument('-fa', '--frozenatoms', type=str, help='Include the \
            the atoms you wish to freeze using the Schrodinger substructure \
            language. Also include the type of coordinate (B, A, T). Use ";" \
            to seperate multiple coordinates. \
            Example: -fa "C2.C2,B;PD-C2.C2,A"')
    parser.add_argument('-d', '--directory', type=str, help='Specify the \
            directory location you would like the gaussian *.com file to be \
            written in.')
    return parser

if __name__ == '__main__':
    main(sys.argv[1:])



