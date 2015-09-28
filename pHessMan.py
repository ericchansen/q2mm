#!/bin/env python

import sys, math, os, json
import numpy as np

# This script reads Hessian information from Jaguar input files
# and performes matrix operations on the (mass-weighted) Hessian
# More file types to be added


def readmass(fm='default.mass'):
    """Read atomic masses from 'default.mass' in $HOME/bin"""
    # fullm=os.environ['HOME']+'/bin/'+fm
    fullm = fm
    with open(fullm,'r') as f:
        i=0
        m=[(0,'Du',0.0,0.0)]
        for lin in f:
            if '*******' in lin:
                i=1
                continue
            elif i==0: continue
            s=lin.split()
            if s[2]=='0': x=s[3]
            elif s[4]=='1': m.append((int(s[0]), s[1].title(), float(s[3]), float(x)))
    return m
 
def readjagin(fn):
  with open(fn,'r') as f:
    if verbose: print "File", fn
    fi=iter(f)   # Chose iter instead of for loop, to allow skipping lines
    natoms=0
    nhess=0
    mc=(0.,0.,0.)
    mt=0.
    atoms=[]     # List of tuples with atomic information
    hessian=[]   # List of lists, Hessian matrix
    while True:
        try:
            lin=fi.next()
        except: break
        curr=lin.split()
        try:
            if curr[0]=='&zmat': # Inside molecular geometry section
                if verbose: print "Found molecular geometry"
                zmat=False
                while True:
                    lin=fi.next()
                    curr=lin.split()
                    if curr[0]=='&': break
                    natoms=natoms+1
                    at=curr[0]
                    at=at[:2]
                    if not at.isalpha(): at=at[:1]
                    at=at.title()
                    m=atmass[at]
                    try: x, y, z = float(curr[1]), float(curr[2]), float(curr[3])
                    except:
                        zmat=True
                        x, y, z = 0., 0., 0. 
                    t=(at, x, y, z, m)
                    atoms.append(t)
                    if verbose: print " {:2s} {:10.6f} {:10.6f} {:10.6f}    m={:-.6f}".format(at, x, y, z, m)
                    mc=(mc[0]+m*x,mc[1]+m*y,mc[2]+m*z)
                    mt=mt+m
                if verbose:
                    print "Found "+repr(natoms)+" atoms"
                    if zmat: print "Cannot understand z-matrix format, faking geometry"
                    else: print "Mass center: "+str((mc[0]/mt,mc[1]/mt,mc[2]/mt))

            elif curr[0]=='&hess': # Inside Hessian section
                if verbose: print "Found Hessian information"
                am=[]
                for at, x, y, z, m in atoms: # Generate mass weighting vector
                    rm=1/math.sqrt(m)
                    am.append(rm)
                    am.append(rm)
                    am.append(rm)
                while True:
                    lin=fi.next()
                    curr=lin.split()
                    if curr[0]=='&': break
                    n=int(curr.pop(0))-1
                    i=len(curr)
                    nhess=nhess+i
                    if i==0:
                        k=n
                        continue
                    if k==0: hessian.append([])
                    j=k
                    for s in curr:
                        x=float(s)*am[n]*am[j]*hessconv
                        hessian[n].append(x)
                        if j!=n: hessian[j].append(x)
                        j=j+1
                if verbose: print "Found "+repr(nhess)+" Hessian elements, "+repr(3*natoms*(3*natoms+1)/2)+" expected"
        except:
            print "Could not understand the file"
            break
  return atoms,hessian

def readjagout(fn):
  with open(fn,'r') as f:
    if verbose: print "File", fn
    fi=iter(f)   # Chose iter instead of for loop, to allow skipping lines
    atoms=[]     # List of tuples with atomic information
    eval=[]      # List of eigenvalues, force constants
    evec=[]      # List of lists, Eigenvector matrix, normal modes
    nfc=[]       # Temporary list, one line of force constants
    nvec=[]      # Temporary list, one set of eigenvectors
    sqm=[]        # sqrt(mass) for coordinate weighting
    ne=0         # Number of eigenvalues/vectors found
    islin=False  # Is the molecule linear?
    while True:
        try:
            lin=fi.next()
        except: break
        if 'geometry:' in lin: #Read in all geometries, in overwrite mode
            natoms=0
            del(atoms[:])
            fi.next()
            fi.next()
            lin=fi.next()
            curr=lin.split()
            while len(curr)>0:
                natoms=natoms+1
                at=curr[0]
                at=at[:2]
                if not at.isalpha(): at=at[:1]
                at=at.title()
                x, y, z, m = float(curr[1]), float(curr[2]), float(curr[3]), atmass[at]
                t=(at, x, y, z, m)
                atoms.append(t)
                x=math.sqrt(m)
                for i in range(3): sqm.append(x)
                lin=fi.next()
                curr=lin.split() 
            if verbose:
                print "Found new geometry, "+repr(natoms)+" atoms"
                for at, x, y, z, m in atoms:
                    print " {:2s} {:10.6f} {:10.6f} {:10.6f}".format(at, x, y, z)
        elif 'frequ' in lin:
            curr=lin.split()
            if 'frequ' in curr.pop(0):
                del(nfc[:]) # Empty temporary lists
                del(nvec[:])
                for s in curr:
                    if float(s)<0.: nfc.append(-1.) # Report imag. frequencies as negative force constants
                    else: nfc.append(1.)
                    nvec.append([])
                    ne=ne+1
                while not 'reduc' in lin: lin=fi.next()
                curr=lin.split()
                for i in range(len(nfc)):
                    nfc[i]=nfc[i]/float(curr[i+2])
                lin=fi.next()
                while not 'force' in lin: lin=fi.next()
                curr=lin.split()
                for i in range(len(nfc)):
                    nfc[i]=nfc[i]*float(curr[i+2])/fcconst
                lin=fi.next()
                curr=lin.split()
                nel=0
                sqmit=iter(sqm)
                while len(curr)>2: # Terminated by blank line
                    x=sqmit.next()
                    for i in range(len(nvec)): nvec[i].append(float(curr[i+2])*x)
                    nel=nel+1
                    lin=fi.next()
                    curr=lin.split()
                for i in range(len(nvec)): # Accumulate Eigensystem
                    eval.append(nfc[i])    # Gives error if number of eigenvalues/vectors differ.
                    evec.append(nvec[i])
        elif 'Molecule is linear' in lin: islin=True
    if verbose:
        print "Found                 : "+repr(ne)+" eigenvectors with "+repr(nel)+" elements."
        if islin: print "Expected for linear   : "+repr(natoms*3-5)+" eigenvectors with "+repr(natoms*3)+" elements."
        else: print "Expected for nonlinear: "+repr(natoms*3-6)+" eigenvectors with "+repr(natoms*3)+" elements."
    return atoms,eval,evec

def readgauout(fn):
  with open(fn,'r') as f:
    if verbose: print "File", fn
    fi=iter(f)   # Chose iter instead of for loop, to allow skipping lines
    atoms=[]     # List of tuples with atomic information
    eval=[]      # List of eigenvalues, force constants
    evec=[]      # List of lists, Eigenvector matrix, normal modes
    nfc=[]       # Temporary list, one line of mass weighted force constants
    nvec=[]      # Temporary list, one set of eigenvectors
    ne=0         # Number of eigenvalues/vectors found
    islin=False  # Is the molecule linear?
    HPMode=False # High precision mode (desired)
    FirstHarm=False
    while True:
        try:
            lin=fi.next()
        except: break
        if 'orientation:' in lin: #Read in all geometries, in overwrite mode
            natoms=0
            del(atoms[:])
            fi.next()
            fi.next()
            fi.next()
            fi.next()
            lin=fi.next()
            curr=lin.split()
            while not '---' in lin:
                natoms=natoms+1
                i=int(curr[1])
                at=mass[i][1]
                x, y, z, m = float(curr[3]), float(curr[4]), float(curr[5]), mass[i][mopt]
                t=(at, x, y, z, m)
                atoms.append(t)
                lin=fi.next()
                curr=lin.split() 
            if verbose:
                print "Found new geometry, "+repr(natoms)+" atoms"
                for at, x, y, z, m in atoms:
                    print " {:2s} {:10.6f} {:10.6f} {:10.6f}".format(at, x, y, z)
        elif 'Harmonic' in lin:
            if FirstHarm: break
            else: FirstHarm=True
        elif 'Frequencies' in lin:
            curr=lin.split()
	    del(nfc[:]) # Empty temporary lists
	    del(nvec[:])
            curr=curr[2:]
	    for s in curr:
		if float(s)<0.: nfc.append(-1.) # Report imag. frequencies as negative force constants
		else: nfc.append(1.)
		nvec.append([])
		ne=ne+1
	    lin=fi.next()
	    curr=lin.split()
	    for i in range(len(nfc)):
		nfc[i]=nfc[i]/float(curr[i+3])
	    lin=fi.next()
	    curr=lin.split()
	    for i in range(len(nfc)):
		nfc[i]=nfc[i]*float(curr[i+3])/fcconst
	    fi.next()
	    lin=fi.next()
            if 'Coord' in lin: HPMode=True
	    lin=fi.next()
	    curr=lin.split()
	    nel=0
            cl=len(curr)
	    while len(curr)==cl:
                if 'Harmonic' in lin: break
                if HPMode:
                    curr=curr[1:]
                    nel=nel+1
                else: nel=nel+3
                # print int(curr[1]), mass[int(curr[1])], mass[int(curr[1])][mopt]
                m=math.sqrt(mass[int(curr[1])][mopt])
                curr=curr[2:]
		for i in range(len(nvec)):
                    if HPMode:
                        a = curr.pop(0)
                        # print a
                        # nvec[i].append(float(curr.pop(0))*m)
                        nvec[i].append(float(a)*m)
                    else: 
                        for j in range(3):
                            a = curr.pop(0)
                            # print a, m
                            nvec[i].append(float(a)*m)
		lin=fi.next()
		curr=lin.split()
	    for i in range(len(nvec)): # Accumulate Eigensystem
		eval.append(nfc[i])    # Gives error if number of eigenvalues/vectors differ.
		evec.append(nvec[i])
            if 'Harmonic' in lin: break
    # print '=' * 50
    # print 'evec: {}'.format(evec)
    for nv in evec:
        # print 'nv: {}'.format(nv)
        # print '~' * 50
        ss=0.
        for x in nv:
            ss=ss+x*x
        x=1/math.sqrt(ss)
        # print 'ss: {}\tx: {}'.format(ss, x)
        for i in range(len(nv)): nv[i]=nv[i]*x
    if verbose:
        print "Found                 : "+repr(ne)+" eigenvectors with "+repr(nel)+" elements."
        print "Expected for nonlinear: "+repr(natoms*3-6)+" eigenvectors with "+repr(natoms*3)+" elements."
    return atoms,eval,evec

def readgauarch(fn):
  with open(fn,'r') as f:
    if verbose: print "File", fn
    alin=""
    ia=False
    for lin in f:
        if '\\' in lin: # Found archive entry
            ia=True 
	    atoms=[]
            hessian=[]
        if ia: alin=alin+lin[1:-1]
        if '@' in lin:
            ait=iter(alin.split('\\'))
            s=ait.next()
            while True:
                try: s=ait.next()
                except: break
                if '#' in s:
                    for i in range(4): s=ait.next()
                    while len(s)>0:
                        a=s.split(',')
                        at=a[0]
                        x,y,z,m=float(a[1]),float(a[2]),float(a[3]),atmass[at]
                        t=at,x,y,z,m
                        atoms.append(t)
			s=ait.next()
                elif "NImag" in s:
		    am=[]
		    for at, x, y, z, m in atoms: # Generate mass weighting vector
			rm=1/math.sqrt(m)
			am.append(rm)
			am.append(rm)
			am.append(rm)
		    s=ait.next()
		    s=ait.next()
		    a=s.split(',')
                    j, k = 0, 0
                    while len(a)>0:
			x=float(a.pop(0))*am[j]*am[k]*hessconv
                        if j==0: hessian.append([])
			hessian[j].append(x)
                        if j==k: j, k = 0, k+1
                        else:
                            hessian[k].append(x)
                            j=j+1 
            ia=False
            alin=""
    return atoms,hessian

def readgaufchk(fn):
  aNums=[]
  cartesians=[]
  aWeights=[]
  cartredmass=[]
  atoms=[]     # List of tuples with atomic information
  hessian=[]
  with open(fn,'r') as f:
    if verbose: print "File", fn
    fi=iter(f)   # Chose iter instead of for loop, to allow skipping lines
    while True:
        try: lin=fi.next()
        except: break
        if 'Atomic numbers' in lin:
            words=lin.split()
            nAtoms=int(words[-1])
            i=0
            while i<nAtoms:
                lin=fi.next()
                words=lin.split()
                for an in words:
                    i=i+1
                    aNums.append(int(an))
        elif 'cartesian coordinates' in lin:
            words=lin.split()
            nCarts=int(words[-1])
            i=0
            while i<nCarts:
                lin=fi.next()
                words=lin.split()
                for cart in words:
                    i=i+1
                    cartesians.append(float(cart)*bohr2angstrom)
        elif 'Real atomic weights' in lin:
            words=lin.split()
            nWeights=int(words[-1])
            i=0
            while i<nWeights:
                lin=fi.next()
                words=lin.split()
                for w in words:
                    i=i+1
                    m=float(w)
		    rm=1/math.sqrt(m)
                    aWeights.append(m)
                    for j in range(3): # Save the reduced masses for each coordinate
                        cartredmass.append(rm)
        elif 'Cartesian Force Constants' in lin:
            words=lin.split()
            nHess=int(words[-1])
            i, j=0, 0
            while i<nCarts:
                lin=fi.next()
                words=lin.split()
                for h in words:
                    helem=float(h)*cartredmass[i]*cartredmass[j]*hessconv
                    if j==0: hessian.append([])
                    hessian[i].append(helem)
                    if j<i: hessian[j].append(helem)
                    j=j+1
                    if j>i:
                        i, j = i+1, 0
    for i in range(nAtoms):
        j=i*3
	t=mass[aNums[i]][1],cartesians[j],cartesians[j+1],cartesians[j+2],aWeights[i]
        i=i+1
	atoms.append(t)
    return atoms,hessian

def readmaclog(fn):
  with open(fn,'r') as f:
    if verbose: print "File", fn
    fi=iter(f)   # Chose iter instead of for loop, to allow skipping lines
    hessian=[]   # List of lists, Hessian matrix
    nhess=0
    while True:
        try:
            lin=fi.next()
        except: break
        if 'Mass-weighted' in lin:
            fi.next()
            lin=fi.next()
            while True:
                if 'Element' in lin:
                    g=1
		    lin=fi.next()
                    he=[]
		    hessian.append(he)
                curr=lin.split()
                if len(curr)==0: break
                for s in curr:
                    g=1-g
                    if g:
                        he.append(float(s))
                        nhess=nhess+1
		lin=fi.next()
    if verbose: print "Found mass-weighted Hessian with "+repr(nhess)+" elements"
    return hessian

def arprint(arr,s="",sstest=False,pstyle="f"):
    t=arr.shape
    ss=0.
    print s
    if len(t)==1:
        for i in range(t[0]): print " {:10.5f}".format(arr[i]),
        print
    elif len(t)==2:
        for i in range(t[0]):
            for j in range(t[1]):
                x=arr[i][j]
                if j<i: ss=ss+x*x
                print " {:10.5f}".format(x),
            print
        if sstest: print "Sqrt(Sum of Squares of Lower triangular off-diagonal): {:10.5f}".format(math.sqrt(ss))
    else: "Cannot print higher dimensional objects"

# Variable initialization

sys.argv.pop(0)            # Remove the script name
mass=readmass()            # Read in a list of tuples with atomic masses
verbose=False
useHess=False              # Try to use ET*H*E to get Eigenvalues
atmass={}
mopt=2
for t in mass: atmass[t[1]]=t[mopt]
fcconst=15.569141    # From au to mdyn/A
hessconv=9375.829222 # Converting Hessian elements from au to MacroModel units
freqconst=53.0883777868  # From Eigenvalues of mass-weighted Hessian to cm-1
bohr2angstrom=0.5291772086
sq2=1/math.sqrt(2.)
label="D1"           # Default label for parametrization
wdia=0.1              # Default parametterization weight for diagonal elements
woff=0.05              # Default parametterization weight for off-diagonal elements

# Check for recognized options and execute appropriate command

argit=iter(sys.argv)
while True:
    try:
        fn=argit.next()
    except: break
    if fn=='-v': verbose=True
    elif fn=='-label':
        try: 
            label=argit.next()
	except: break
    elif fn=='-wt':
        try:
            x=argit.next().split(',')
            wdia=float(x[0])
        except:
            print "Invalid weight, quitting"
            exit()
        print x
        try: woff=float(x[1])
        except: woff=wdia/2
        if verbose: print "New scaling factors, diagonal elements: {0:.4f}; offdiagonal elements: {1:.4f}".format(wdia,woff)
    elif fn=='-ab':
        mopt=2
        for t in mass: atmass[t[1]]=t[mopt]
        if verbose: print "Most abundant isotope mass used"
    elif fn=='-av':
        mopt=3
        for t in mass: atmass[t[1]]=t[mopt]
        if verbose: print "Natural average  isotope mass used"
    elif fn=='-rgout':
        try: 
            fn=argit.next()
        except: break
	atoms,fcs,eigvec=readgauout(fn)
	if verbose:
	    print ">>> Finished reading Gaussian output file "+fn
	    print "Geometry:"
	    for at,x,y,z,m in atoms: print " {:2s} {:9.6f} {:9.6f} {:9.6f}  m={:-9.6f}".format(at,x,y,z,m)
	    print "Eigenvalues:"
	    for x in fcs: print " {:12.3E}".format(x),
	    print
	    print "Mass-weighted eigenvectors:"
	    for i in range(len(eigvec[0])):
		for j in range(len(eigvec)): print " {:12.3E}".format(eigvec[j][i]),
		print
    elif fn=='-rgarch':
        try: 
            fn=argit.next()
        except: break
	atoms,hessian=readgauarch(fn)
	if verbose:
	    print ">>> Finished reading Gaussian archive from file "+fn
	    print "Geometry:"
	    for at,x,y,z,m in atoms: print " {:2s} {:9.6f} {:9.6f} {:9.6f}  m={:-9.6f}".format(at,x,y,z,m)
	    print "Mass-weighted hessian:"
	    for h in hessian:
		for x in h: print " {:10.3E}".format(x),
		print
    elif fn=='-rgfchk':
        try:
            fn=argit.next()
        except: break
	atoms,hessian=readgaufchk(fn)
        if verbose:
            print ">>> Finished reading Gaussian formatted checkpoint file "+fn
            print "Geometry:"
            for at,x,y,z,m in atoms: print " {:2s} {:9.6f} {:9.6f} {:9.6f}  m={:-9.6f}".format(at,x,y,z,m)
	    print "Mass-weighted hessian:"
	    for h in hessian:
		for x in h: print " {:10.3E}".format(x),
		print
    elif fn=='-rjout':
        try: 
            fn=argit.next()
	except: break
	atoms,fcs,eigvec=readjagout(fn)
	if verbose:
	    print ">>> Finished reading Jaguar output file "+fn
	    print "Geometry:"
	    for at,x,y,z,m in atoms: print " {:2s} {:9.6f} {:9.6f} {:9.6f}  m={:-9.6f}".format(at,x,y,z,m)
	    print "Eigenvalues:"
	    for x in fcs: print " {:12.3E}".format(x),
	    print
	    print "Mass-weighted eigenvectors:"
	    for i in range(len(eigvec[0])):
		for j in range(len(eigvec)): print " {:12.5E}".format(eigvec[j][i]),
		print
    elif fn=='-rjin':
        try: 
            fn=argit.next()
        except: break
	atoms,hessian=readjagin(fn)
	if verbose:
	    print ">>> Finished reading Jaguar input file "+fn
	    print "Geometry:"
	    for at,x,y,z,m in atoms: print " {:2s} {:9.6f} {:9.6f} {:9.6f}".format(at,x,y,z)
	    print "Mass-weighted hessian:"
	    for h in hessian:
		for x in h: print " {:10.3E}".format(x),
		print
    elif fn=='-mlog': # Read mass-weighted Hessian from MacroModel log
        try: 
            fn=argit.next()
        except: break
	hessian=readmaclog(fn)
	if verbose:
	    print ">>> Finished reading MacroModel log-file "+fn
	    print "Mass-weighted hessian:"
	    for h in hessian:
		for x in h: print " {:12.5E}".format(x),
		print
    elif fn=='-eht': # Multiply Hessian w. Eigenvectors from both sides
        ex=0
        try:
            ex,ey=len(eigvec),len(eigvec[0])
            if verbose:
                print "Eigenvector length: ", repr(ey)
                print "Number of eigenvectors: ", repr(ex)
	    ae=np.array(eigvec)
	    ar=np.dot(ae,ae.T)
            ss=0.
	    print "E_T * E"
	    for i in range(ex):
                x=ar[i][i]
                y=x-1.
                ss=ss+y*y-x*x
		for j in range(ex):
		   print "{:10.6f}".format(ar[j][i]),
                   x=ar[j][i]
                   ss=ss+x*x
		print 
            print "Deviation from orthonormality: {:-12.5E}".format(math.sqrt(ss))
        except: print "Invalid Eigenvectors"
        if ey>0:
            try:
		hx,hy=len(hessian),len(hessian[0])
		if hx!=hy:
		    print "Non-symmetric Hessian!"
		    exit()
		if hx!=ey:
		    print "Incompatible size of eigenvector"
		    exit()
		if verbose:
		    print "Hessian height & width: ",repr(hy),"x", repr(hx)
		ah=np.array(hessian)
		ar=np.dot(np.dot(ae,ah),ae.T)
		print "E_T * H * E"
                ss=0.
                hdia=[]
		for i in range(ex):
                    hdia.append(ar[i][i])
		    for j in range(ex):
                       x=ar[j][i]
		       print "{:10.6f}".format(x),
                       if j<i: ss=ss+x*x
		    print 
		print "Sqrt(sum of squares) of off-diagonal elements: {:-12.5E}".format(math.sqrt(ss))
                print "Frequencies (cm-1):",
                for x in hdia:
                    if x<0.:y=-math.sqrt(-x)
                    else: y=math.sqrt(x)
                    print "{:-.2f}".format(y*freqconst),
                print
	    except: print "No valid Hessian"
    elif fn=='-eiginv': # If Eigenvectors are present, use them, otherwise diagonalize Hessian
                        # Modify most negative eigenvalue, recreate Hessian
        try: ah=np.array(hessian)
        except:
            print "No valid Hessian"
            exit()
	if verbose: arprint(ah, "Hessian, H")
	try: ae=np.array(eigvec)
	except: ae=np.array([])
	if len(ae)<1: 
            w,v=np.linalg.eigh(ah)
            ad=np.diag(w)
            ae=v.T
            if verbose: print "Created non-projected Eigensystem"
            eigvec=[]
	    for i in range(len(ae)):
                eigvec.append([])
		for j in range(len(ae)):
		    eigvec[i].append(ae[i][j])
        else:
            ad=np.dot(np.dot(ae,ah),ae.T)
            w=np.diagonal(ad).copy()
	ar=np.dot(np.dot(ae.T,ad),ae)
	if verbose:
            arprint(w, "Diagonal")
            arprint(ad, "Diagonal matrix",True)
            arprint(ar, "Recreated Hessian")
            arprint(np.subtract(ah,ar),"Diff",True)
	min=0.
	mini=-1
	for i in range(len(w)):
	    x=w[i]
	    if x<min: min,mini=x,i
        if mini<0:
            print "No negative Eigenvalue found"
            break
	w[mini]=hessconv
	ai=np.diag(w)
	ar=np.dot(np.dot(ae.T,ai),ae)
        for i in range(len(ar)):
            for j in range(len(ar)):
                hessian[i][j]=ar[i][j]
        if verbose:
            arprint(w,"Modified Diagonal")
            arprint(ar, "Modified Hessian, M")
            arprint(np.subtract(ah,ar),"Diff",True)
    elif fn=='-diag': # Diagnostic printing of internal data
        try:
            print "Diagnostic printing of current internal data"
            ah=np.array(hessian)
            arprint(ah, "Hessian, H")
	    ae=np.array(eigvec)
            if len(ae)==len(ah): arprint(ae.T,"Non-projected, mass-weighted Eigenvectors, E")
	    else: arprint(ae.T,"Projected mass-weighted Eigenvectors, E")
	    arprint(np.dot(np.dot(ae,ah),ae.T),"E * H * E_T",True) 
        except: print "Could not find Eigenvectors and/or Hessian"
    elif "-dump" in fn: # General save command, currently only supported for -dumpeig
        fw="Unknown"
        try: 
            fw=argit.next()
	except:
            print "Not valid file, "+fw+", quitting"
            break
	with open(fw,'w') as f:
	    if verbose: print "Opened file "+fw+" for writing"
	    if "eig" in fn: json.dump(eigvec,f)
    elif "-load" in fn: # General read command, currently only supported for -loadeig
        fr="Unknown"
        try: 
            fr=argit.next()
	except:
            print "Not valid file, "+fr+", quitting"
            break
	with open(fr,'r') as f:
	    if verbose: print "Opened file "+fr+" for reading"
	    if "eig" in fn: eigvec=json.load(f)
            if verbose:
		print "Mass-weighted eigenvectors:"
		for i in range(len(eigvec[0])):
		    for j in range(len(eigvec)): print " {:12.5E}".format(eigvec[j][i]),
		    print
    elif fn=='-uhess': useHess=True # Use ET*H*E for Eigenvalues
    elif fn=='-wparref': # Write Eigenvalues and off-diagonals as parameterization reference data
        fw="Unknown"
        MissEigVal=True
        try: 
            fw=argit.next()
	except:
            print "Not valid file, "+fw+", quitting"
            break
        try:
            ar=np.diag(np.array([x*hessconv for x in fcs]))
            MissEigVal=False
        except:
            print "Did not find valid Eigenvalues, trying for Hessian data"
        if MissEigVal or useHess:
	    try:
		ae=np.array(eigvec)
		ah=np.array(hessian)
		ar=np.dot(np.dot(ae,ah),ae.T)
		MissEigVal=False
	    except:
		print "Did not find valid Hessian and/or Eigenvectors"
                exit()
	with open(fw,'w') as f:
	    if verbose:
                print "Opened file "+fw+" for writing"
                print "Writing E * H * E_T in lower triangular format"
	    for i in range(len(ar)):
                for j in range(i):
                    f.write("{}_{}_{}\t{:.4f}\t{:10.4f}\n".format(label,i+1,j+1,woff,0.))
		f.write("{}_{}_{}\t{:.4f}\t{:10.4f}\n".format(label,i+1,i+1,wdia,ar[i][i]))
    elif fn=='-wparone': # Write Eigenvalues and off-diagonals as parameterization reference data
        fw="Unknown"
        try: 
            fw=argit.next()
	except:
            print "Not valid file, "+fw+", quitting"
            break
        try:
	    ae=np.array(eigvec)
            ah=np.array(hessian)
            ar=np.dot(np.dot(ae,ah),ae.T)
        except:
            print "Did not find valid Hessian and/or Eigenvectors"
            exit()
	with open(fw,'w') as f:
	    if verbose:
                print "Opened file "+fw+" for writing"
                print "Writing E * H * E_T in lower triangular format"
	    for i in range(len(ar)):
                for j in range(i+1):
                    f.write("{:10.4f}\n".format(ar[i][j]))
    else:
        print "Unrecognized option: "+fn

