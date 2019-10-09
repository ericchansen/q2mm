import openbabel as ob
import pybel as pb
import numpy as np
import glob, os
import multiprocessing as mp
from multiprocessing import Pool


def bmin_call(file_list):
    for filename in file_list:
        infile = filename.replace(".com","")
        os.system("bmin -WAIT {};".format(infile))
    return 0



def cons_opt(infile,outfile,idx):
    """
    Only Mol2/UFF can Optimize
    Optimize with Pd1-P-P1 constraints
    :param infile:
    :param outfile:
    :param idx:
    :return:
    """
    p1,p2,pd = idx

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("mol2", "mol2")
    mol = ob.OBMol()
    conv.ReadFile(mol, infile)

    cons = ob.OBFFConstraints()
    pp = 3.2
    ppd = 2.4
    cons.AddDistanceConstraint(p1, p2, pp)
    cons.AddDistanceConstraint(p1, pd, ppd)
    cons.AddDistanceConstraint(p2, pd, ppd)

        

    # Set up FF
    ff = ob.OBForceField.FindForceField("UFF")
    ff.Setup(mol, cons)
    ff.SetConstraints(cons)
    ff.EnableCutOff(True)

    # Optimize
    ff.ConjugateGradients(10000)
    ff.GetCoordinates(mol)


    def ring_bond(ring):
        """

        :param ring: OBRing class
        :return: list of lists [atom1,atom2]
        """
        bonds = []
        mol = ring.GetParent()
        for bond in ob.OBMolBondIter(mol):
            at1 = bond.GetBeginAtom().GetIndex() + 1
            at2 = bond.GetEndAtom().GetIndex() + 1
            if ring.IsMember(bond):
                if not bond.IsAromatic():
                    bonds.append(sorted([at1,at2]))
        return bonds
    def common_atom(bond,bonds):
        """
        :param bond: list [atom1,atom2]
        :param bonds: list of list [atom1,atom2]
        :return: True if there is common atom in bonds
        """
        result = False
        if len(bonds) == 0:
            return result
        for bond2 in bonds:
            for at1 in bond:
                for at2 in bond2:
                    if at1 == at2:
                        result = True
                        return result
        return result

    # extract ring info
    # iterate over all bond
    # CHIRAL ATOMS
    chiral = []
    for atom in ob.OBMolAtomIter(mol):
        if atom.IsChiral():
            chiral.append(atom.GetIndex() + 1)
    rot_bond = []
    rca4 = []

    rca23 = []
    for bond in ob.OBMolBondIter(mol):
        at1 = bond.GetBeginAtom().GetIndex() + 1
        at2 = bond.GetEndAtom().GetIndex() + 1
#        print(at1, at2)
        if bond.IsRotor() and not bond.IsAromatic():
            rot = sorted([at1,at2])
#            print(outfile,"ROT ",rot)
            rot_bond.append(rot)
        if bond.IsClosure():
            rca0 = sorted([at1,at2])
            rca = sorted([at1,at2])

            # The Assumption is IsClosure picking up only one bond in the ring
            # and bond.IsRotor() does not provide any ring bonds.
            # to prevent rca23 sharing common atoms
            if len(rca23) != 0:
                if common_atom(rca,rca23):
                    ringbonds = ring_bond(bond.FindSmallestRing())
                    for rbond in ringbonds:
                        if common_atom(rbond,rca23):
                            continue
                        else:
                            rca0 = rbond.copy()
                            rca = rbond.copy()
                            break
            else:
                ringbonds = ring_bond(bond.FindSmallestRing())
                for rbond in ringbonds:
                    if not (rbond[0] in rca or rbond[1] in rca):
                        rca0 = rbond.copy()
                        rca = rbond.copy()
                        break
            rca23.append(rca0)

            #print(outfile,"RING OPENING BOND", rca0)
            ring = bond.FindSmallestRing()
            ring_rots = []
            if not ring.IsAromatic():
                for bond1 in ob.OBMolBondIter(mol):
                    if ring.IsMember(bond1):
                        b1 = bond1.GetBeginAtom().GetIndex() + 1
                        b2 = bond1.GetEndAtom().GetIndex() + 1
                        rot = sorted([b1,b2])
                        if rot != rca0:
                            ring_rots.append(rot)
                #print("RING ROT:",ring_rots)
                for rrot in ring_rots:
                    if rca0[0] in rrot:
                        atom = rrot.copy()
                        atom.remove(rca0[0])
                        #print(atom)
                        rca.insert(0,atom[0])
                        #print("INSERT ",rca)
                    elif rca0[1] in rrot:
                        atom = rrot.copy()
                        atom.remove(rca0[1])
                        #print(rca)
                        rca.append(atom[0])
                        #print("APPEND ",rca)
                    elif rca0 != rot:
                        rot_bond.append(rrot)
                    
                rca4.append(rca)
    # print(outfile,"CHIRAL",chiral)
    # print(outfile,"RCA4",rca4)
    # print(outfile,"RCA23",rca23)
    check = []
    for rr in rca4:
        check.append(rr[1])
        check.append(rr[2])
        bond = sorted([rr[1],rr[2]])
        if bond in rot_bond:
            rot_bond.remove(bond)
    if len(check) != len(set(check)):
        print("\t\tBAD",outfile)
        print("\t\t",rca4)
 #   else:
#        print("\t\tBAD",outfile)

    # print(outfile,"ROT", rot_bond)

    conv.WriteFile(mol, outfile)
    return chiral, rca4, rot_bond

def to_mol2(infile,outfile):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("sdf","mol2")
    mol = ob.OBMol()
    conv.ReadFile(mol, infile)
    conv.WriteFile(mol, outfile)

    
def add_pd(infile,outfile):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("mol2","mol2")
    mol = ob.OBMol()
    conv.ReadFile(mol, infile)

    nAtoms = mol.NumAtoms()

    p = []
    for i in range(nAtoms):
        n = i + 1
        at = mol.GetAtom(n)
        an = (at.GetAtomicNum())
        if an == 15:
            p.append(n)
        elif an == 8:
            # Oxygen
            neis = []
            for nei in ob.OBAtomAtomIter(at):
                neis.append(nei)
            # Oxygen has one neighbor
            lnei = len(neis)
            if lnei == 1:
                nei = neis[0]
                # neighbor is P (i.e. P=O)
                if nei.GetAtomicNum() == 15:
                    return None

    if len(p) != 2:
        return None
    # optimize for P-P distance first
    p1, p2 = p
    cons = ob.OBFFConstraints()
    pp = 3.2
    cons.AddDistanceConstraint(p1, p2, pp)

    # Set up FF
    ff = ob.OBForceField.FindForceField("UFF")
    ff.Setup(mol, cons)
    ff.SetConstraints(cons)
    ff.EnableCutOff(True)
    # Optimize
    ff.ConjugateGradients(10000)
    ff.GetCoordinates(mol)
    cont = True
    while cont:
        pho1 = mol.GetAtom(p1)
        pho2 = mol.GetAtom(p2)
        pp1 = pho1.GetDistance(pho2)
        err0 = abs(pp1-pp)
        if err0 < 0.015:
            cont = False

        else:
            print("\tNOT converged YET:",outfile, " diff:", err0)
            ff.ConjugateGradients(10000)
            ff.GetCoordinates(mol)


    p = []
    pxyz = []
    nxyz = []
    # find out where two P are located
    for i in range(nAtoms):
        n = i + 1
        at = mol.GetAtom(n)
        an = (at.GetAtomicNum())
        if an == 15:
            p.append(n)
            pxyz.append([at.x(),at.y(),at.z()])
        else:
            nxyz.append([at.x(),at.y(),at.z()])
    nxyz = np.array(nxyz)






    # Add Pd and connect it to two Ps
    a = mol.NewAtom()
    a.SetAtomicNum(46)
    pxyz = np.array(pxyz)
    x,y,z = (pxyz[0] + pxyz[1])/2
    pdxyz = np.array([x,y,z])
    vec0 = None
    r0 = 100.0
    for vec in nxyz:
        vec = vec - pdxyz
        r = np.linalg.norm(vec)
        if r < r0:
            r0 = r
            vec0 = vec
    x,y,z = pdxyz-10.0*vec0
    a.SetVector(x,y,z)
    # AddBond(BeginIdx,EndIdx,bond order)
    pd = mol.NumAtoms()
    p1,p2 = p
    mol.AddBond(pd,p1,1)
    mol.AddBond(pd,p2,1)
    mol.NumAtoms()
    
    conv.WriteFile(mol, outfile)
    return [p1,p2,pd]

def sdf_to_mae(filenames):
    for fn0 in filenames:
        fn = fn0.replace(".sdf","")
        # print("reading {}".format(fn))
        to_mol2(fn0,"{}.0temp".format(fn))
        index = add_pd("{}.0temp".format(fn),"{}.temp".format(fn))
        if index != None:
            chiral, rca4, rot = cons_opt("{}.temp".format(fn),"{}.mol2".format(fn),index)
            print("molecule ",fn, "finished")
            os.system("mv {}.mol2 ../mol2/".format(fn))
            os.system("mol2convert -imol2 ../mol2/{}.mol2 -omae ../maes/{}.mae".format(fn,fn))
            os.system("echo 'CHIRAL {}\nRCA4 {}\n ROT {}' >> ../maes/{}.mae".format(chiral, rca4,rot,fn))

    return 0

# Get the list of sdf files in current directory
# for loop all
os.system("mkdir ./mol2")
os.system("mkdir ./maes")
os.chdir("./sdf")

if 1:
    fns = glob.glob("*.sdf")
    # number of threads for parallel job
    nt = 32
    lf = len(fns)
    nj = int(lf/nt)
    nr = int(lf%nt)
    ni = 0
    count = 0
    file_split=[]
    for i in range(nt-1):
        fn = fns[ni:ni+nj]
        file_split.append(fn)
        count += len(fn)
        ni = ni + nj
    for i in range(nr):
        file_split[i].append(fns[ni+i])
    with Pool(processes=nt) as pool:
        multiple_jobs = [pool.apply_async(sdf_to_mae,(files,)) for files in file_split]
        [res.get() for res in multiple_jobs]
    
