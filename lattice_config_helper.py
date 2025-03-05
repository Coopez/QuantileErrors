def ceil(a, b):
    return -(a // -b)
input_size = 512
quantile_size = 1
input_per_lattice_per_layer = [3, 3, 3, 3, 3, 3]
output_size = 90
keypoints = 10

complexity = [ceil((input_size + quantile_size),input_per_lattice) * keypoints ** (input_per_lattice) for input_per_lattice in input_per_lattice_per_layer]
total_complexity = sum(complexity)

print(total_complexity)
print(complexity)