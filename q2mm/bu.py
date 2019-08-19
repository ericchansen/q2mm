sander -O -i min.in -o min.out -p parm7 -c rst7 -r traj.ncrst
    """
    1 reference file only
    
    """
    """
    Step 0: run tleap to generate parm7 and rst7
    it sources leaprc file (this may be where the modification may need to be added)
    frcmod.FFname (on current directory)
    tleap -f leap.in
    
    """
    
    """
    Step 1: run minimization (maxcyle = 0 or 500)
    Step 2: run trajectory to produce .nc file (ouput: .nc file)
    Step 3: cpptraj to printout all bond, angle, dihedral (input: parm7, output: angle.in/bond.in)
    Step 4: writeout the current value (input: parm7 .nc file angle.in)
    """
    
    """
    cpptraj prints info of bond, angle, dihedral
    for Force Constant and equilibrium value and atom number
    """
    """
    tleap program to generate the following 
    
    source leaprc.protein.ff14SB # force parameter file here
    foo = sequence { ACE ALA NME }
    source leaprc.water.tip3p
    solvatebox foo TIP3PBOX 10.0
    saveamberparm foo parm7 rst7
    quit
    
    * no need for input if it is Nucleic acid
    parm    = topology file
    http://ambermd.org/formats.html
    parameter value are included in here
    
    
    %FLAG BOND_FORCE_CONSTANT
    FORMAT (RK)
    index? 
    
    %FLAG BOND_
    FORMAT (REQ)
    
    %FLAG BONDS_WITHOUT_HYDROGEN
    %FORMAT
        A B index(for RK REQ) A B index
    atom index starts at 0
