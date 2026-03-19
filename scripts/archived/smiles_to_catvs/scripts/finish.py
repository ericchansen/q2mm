#!/usr/bin/env $SCHRODINGER/run

from schrodinger.application.macromodel.input import *
import schrodinger.application.macromodel.utils as mmutils
import schrodinger.structure as struct
import schrodinger.structutils.analyze as ana
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool


def bmin_call(file_list):
    for filename in file_list:
        infile = filename.replace(".com", "")
        os.system(f"bmin -WAIT {infile};")
    return 0


# Split result.mae to individual mae files
if 1:
    sts = struct.StructureReader("result.mae")
    os.system("mkdir many")
    # os.system("cp atom.typ many/")
    # os.system("cp mm3.fld many/")
    for i, st in enumerate(sts):
        n = i + 1
        st.write(f"temp{n}.mae", format="maestro")
        os.system(f"mv temp{n}.mae many/{n}.mae")
if 1:
    fns = glob.glob("many/*.mae")
    for fn in fns:
        fn0 = fn.replace(".mae", "")
        os.system(f"$SCHRODINGER/run $Q2MM/screen/setup_com_from_mae.py {fn0}.mae {fn0}_cs.com {fn0}_cs.mae")
if 1:
    # for debug
    #    os.chdir("./many")
    fns = glob.glob("many/*.com")
    np = 7
    lf = len(fns)
    nj = int(lf / np)
    nr = int(lf % np)
    ni = 0
    count = 0
    file_split = []
    for i in range(np - 1):
        fn = fns[ni : ni + nj]
        file_split.append(fn)
        count += len(fn)
        ni = ni + nj
    for i in range(nr):
        file_split[i].append(fns[ni + i])
    count += len(fn)
    with Pool(processes=np) as pool:
        multiple_jobs = [pool.apply_async(bmin_call, (files,)) for files in file_split]
        [res.get() for res in multiple_jobs]
