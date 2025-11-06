from ..gravitree import *
from ..ICs import generate_plummer_initial_conditions

def test_gravitree():
    """
    tests the gravitree function by generating a tree for a plummer sphere and checking root com is correct, which requires that all com leading up were additionally correct
    """
    blackholes = generate_plummer_initial_conditions(100, 10, 10)[0]['data']
    com = np.array([
        np.sum([bh.position[0]*bh.mass for bh in blackholes]),
        np.sum([bh.position[1]*bh.mass for bh in blackholes]),
        np.sum([bh.position[2]*bh.mass for bh in blackholes]),
    ])
    com /= np.sum([bh.mass for bh in blackholes]) 
    root = build_tree(blackholes)

    print(root)

    assert (
        np.isclose(com[0], root.center_of_mass[0]) and
        np.isclose(com[1], root.center_of_mass[1]) and
        np.isclose(com[2], root.center_of_mass[2])
    ), "center of mass calculation in gravitree is inaccurate"

    assert (
        np.isclose(blackholes[0].displacement(root)[0], (root.center_of_mass - blackholes[0].position)[0]) and
        np.isclose(blackholes[0].displacement(root)[1], (root.center_of_mass - blackholes[0].position)[1]) and
        np.isclose(blackholes[0].displacement(root)[2], (root.center_of_mass - blackholes[0].position)[2])
    ), "Displacements between nodes and black holes are not being calculated correctly"