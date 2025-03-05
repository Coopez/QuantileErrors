def _ceil(a, b):
    return -(a // -b)
def lattice_layerer(input_dim, input_dim_per_lattice,lattices=[]):
    if input_dim == 1.0:
        return lattices
    inp = _ceil(input_dim, input_dim_per_lattice)
    lattices.append(inp)
    return lattice_layerer(inp, input_dim_per_lattice, lattices)
    




input_dim = 512
input_dim_per_lattice = 3
lattices = lattice_layerer(input_dim, input_dim_per_lattice)

print(lattices)