import pubchempy as pcp
import requests as rq
import sys, os, glob


"""
Combined python script for 
"""

def smi_to_sdf(SMILES):
    pre_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/"
    post_url = "/record/SDF/?record_type=3d&response_type=display"

    results = pcp.get_compounds(SMILES,"smiles")#,record_type="3d")
    for c in results:
        CID = str(c.cid)
        print(c.iupac_name)
        if c.cid == None:
            print(SMILES)
        url = pre_url + CID + post_url
        r = rq.get(url)
        if r.status_code == 200:
            content = r.content
            with open(CID+".sdf","wb") as f:
                f.write(content)
            os.system("mv {}.sdf ../sdf/".format(CID))


# SMILES to SDF files
if 1:
    os.system("mkdir sdf")
    os.chdir("./smi")
    for filename in glob.glob("*.smi"):
        ref = open(filename,"r")
        lines = ref.readlines()
        for line in lines:
            smi_to_sdf(line)
    os.chdir("./..")
