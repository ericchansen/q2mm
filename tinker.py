import argparse                                                                 
import glob                                                                     
import itertools                                                                
import logging                                                                  
import logging.config                                                           
import numpy as np                                                              
import os                                                                       
import random                                                                   
import sys                                                                      
                                                                                
import calculate                                                                
import compare                                                                  
import constants as co                                                          
import datatypes                                                                
import gradient                                                                 
import opt                                                                      
import parameters                                                               
import simplex         


class File(object):
    """
    Base for every other filetype class.
    """
    def __init__(self, path):
        self._lines = None
        self.path = os.path.abspath(path)
        # self.path = path
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]
    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, 'r') as f:
                self._lines = f.readlines()
        return self._lines
    def write(self, path, lines=None):
        if lines is None:
            lines = self.lines
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)

class Tinker(File):
    def __init__(self, path):
        super(Tinker, self).__init__(path)
        self._structures = None
        self.commands = None
        self.name = os.path.splitext(self.filename)[0]
        self.name_log = self.name + 'q2mm.log'
    @property
    def structures(self):
        if self._structures is None: 
            struct = Structure()
            self._structures = [struct]
            with open(filename, 'r') as f:
                f = f.split()
                for line in f:
                    line = line.split()
                    if len(line) == 2:
                        struct.props['total atoms'] = line[0]
                        struct.props['title'] = line[1]
                    if len(line) > 2:
                        indx, ele, x, y, z, at, bonded_atom = line[0], line[1], 
                            line[2], line[3], line[4], line[5], 
                            line[6:].split()
                        struct.atoms.append(Atom(index=indx, element=ele, 
                            x=float(x), y=float(y), z=float(z), atom_type=at,
                            bonded_atom_indices=bonded_atom))
    def get_com_opts(self):
        com_opts = 

                        

FF_INTERACTIONS = "analyze {} -k {} D >> {}"
FF_OPT = "minimize {} -k {} 0.01"
FF_VIB = "vibrate {} -k {} All >> {}"















