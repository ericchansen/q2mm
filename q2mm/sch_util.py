try:
    import schrodinger.structure as struct
    import schrodinger.structutils.analyze as ana
except ImportError:
    struct = None
    ana = None

from q2mm import constants as co


def _require_schrodinger():
    if struct is None:
        raise ImportError(
            "Schrödinger Python API is required for sch_util functions. "
            "Install Schrödinger Suite or use the schrod_indep modules instead."
        )


def interation234(filename):
    """
    Takes mae filename and export the information on 1-2,1-3,1-4 interacting atoms
    This is needed to correctly assign the weights on the Hessian matrix
    :param filename: .mae filename
    :return: tuples of pair of atoms with 1-2,1-3,1-4 interaction
    """
    _require_schrodinger()
    st = next(struct.StructureReader(filename))
    bonds = ana.bond_iterator(st)
    angles = ana.angle_iterator(st)
    tors = ana.torsion_iterator(st)
    int2 = []
    int3 = []
    int4 = []
    for bond in bonds:
        int2.append([bond[0],bond[-1]])
    for angle in angles:
        int3.append([angle[0],angle[-1]])
    for tor in tors:
        int4.append([tor[0],tor[-1]])
    return [int2,int3,int4]

def wht(at1,at2,ints):
    """
    Assign weights for Hessian matrix given 1-2,1-3,1-4 interactions

    :param at1: first atom
    :param at2: second atom
    :param ints: tuples of 1-2,1-3,1-4 interacting pair
    :return: 0 for diagonal, assigned weight constant for all other
    """
    apair = [at1,at2]
    int2, int3, int4 = ints
    if at1 == at2:
        return 0.0
    elif apair in int2:
        return co.WEIGHTS['h12']
    elif apair in int3:
        return co.WEIGHTS['h13']
    elif apair in int4:
        return co.WEIGHTS['h14']
    else:
        return co.WEIGHTS['h']
