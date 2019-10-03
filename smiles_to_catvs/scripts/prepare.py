import os,glob
from multiprocessing import Pool
import pubchempy as pcp
import requests as rq
import ast

def CID_to_IUPAC(cid):
    pre_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/"
    post_url = "/record/SDF/?record_type=3d&response_type=display"

    c = pcp.Compound.from_cid(cid)
    name = c.iupac_name
    if name == None:
        return None
    else:
        return name

center="""  P3-Z0-P3 
  1
"""
txt1 =""" s_cs_pattern
 b_cs_use_substructure
"""
txt2 ="""  b_cs_comp
  b_cs_chig
"""
txt3 = """  b_cs_tors
  i_cs_rca4_1
  i_cs_rca4_2
  r_cs_torc_a5
  r_cs_torc_a6
  r_cs_torc_b5
  r_cs_torc_b6
  i_cs_torc_a1
  i_cs_torc_a4
  i_cs_torc_b1
  i_cs_torc_b4
"""

def prepare_mae_for_screen(file,s_txt):
    CID = file.replace(".mae","")
    infile = open(file,"r")
    flines = infile.readlines()
    output = ""
    count = 0
    b_cs = 0
    rca4 = None
    rot = None
    chiral = None
    for line in flines:
        if CID in line:
            iupac = CID_to_IUPAC(CID)
            if iupac == None:
                output += line
            else:
#                output += iupac + "\n"
                output += line.replace(CID,iupac.replace(" ",""))
        elif "i_m_ct_format" in line:
            output += line + txt1
        elif "m_atom" in line and count == 0:
            output += s_txt + line
            count = 1
        elif " PD " in line:
            numb = line.split()[1]
            output += line.replace(" {} ".format(numb)," 62 ")
        elif "CHIRAL" in line:
            chiral = "".join(line.split()[1:])
            chiral = ast.literal_eval(chiral)
        elif "RCA4" in line:
            rca4 = "".join(line.split()[1:])
            rca4 = ast.literal_eval(rca4)
        elif "ROT" in line:
            rot = "".join(line.split()[1:])
            rot = ast.literal_eval(rot)
        else:
            output += line
    old_out = output
    output = ""
    count = 0
    for line in old_out.splitlines():
        if "m_atom" in line:
            count = 1
            output += line
        elif count == 1 and ":::" in line:
            count = 2
            output += txt2 + line
        # atom flag
        elif count == 2 and ":::" in line:
            count = 3
            output += line
        # bond flag
        elif count == 3 and ":::" in line:
            count = 4
            output += txt3 + line
        # turn off
        elif count == 4 and ":::" in line:
            count = 5
            output += line
        # atom
        elif count == 2:
            chi = 0
            atidx = int(line.split()[0])
            if atidx in chiral:
                chi = 1
            comp = 1
            if "H " in line:
                comp = 0
            output += line + " {} {}".format(comp,chi)
        # bond
        elif count == 4:
            bond = list(map(int,line.split()[1:3]))
            tors = 0
            if bond in rot or bond[::-1] in rot:
                tors = 1
            rca4_1 = 0
            rca4_2 = 0
            for rca in rca4:
                if bond == rca[1:3]:
                    rca4_1 = rca[0]
                    rca4_2 = rca[3]
                elif bond[::-1] == rca[1:3]:
                    rca4_1 = rca[3]
                    rca4_2 = rca[0]

            output += line + " {} {} {} 0 0 0 0 0 0 0 0".format(tors,rca4_1,rca4_2)
        else:
            output += line
        output += "\n"
    return output





os.system("mkdir ligands")
os.chdir("./maes")
if 1:
    for nfile, file in enumerate(glob.glob("*.mae")):
        n = nfile + 1
        print(file)
        text = prepare_mae_for_screen(file,center)
        outfile =  open("temp","w")
        outfile.write(text)
        outfile.close()
        os.system("cp temp ../ligands/{}.mae".format(str(n)))

def prepare(filenames):
    for fn0 in filenames:
        text = prepare_mae_for_screen(fn0,center)
        
        fn = fn0.replace(".mae","")
        temp = "{}.temp".format(fn)
        outfile = open(temp,"w")
        outfile.write(text)
        outfile.close()
        os.system("mv {} ../ligands/{}.mae".format(fn,fn))

    return 0
if 0:
    fns = glob.glob("*.mae")
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
        multiple_jobs = [pool.apply_async(prepare,(files,)) for files in file_split]
        [res.get() for res in multiple_jobs]
