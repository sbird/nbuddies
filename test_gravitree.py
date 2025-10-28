from gravitree import *

def test_gravitree():
    """
    tests the gravitree function by generating a tree for a plummer sphere and checking root com is correct, which requires that all com leading up were additionally correct
    """
    blackholes = generate_plummer_initial_conditions(20, 10, 10)[0]
    com = np.array([
        np.sum([bh.position[0]*bh.mass for bh in blackholes]),
        np.sum([bh.position[1]*bh.mass for bh in blackholes]),
        np.sum([bh.position[2]*bh.mass for bh in blackholes]),
    ])
    com /= np.sum([bh.mass for bh in blackholes]) 
    root = build_tree(blackholes)

    assert (
        np.isclose(com[0], root.center_of_mass[0]) and
        np.isclose(com[1], root.center_of_mass[1]) and
        np.isclose(com[2], root.center_of_mass[2])
    ), "center of mass calculation in gravitree is inaccurate"
