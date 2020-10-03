from src.mcla.mcla import mcla

comm = [
    {0, 1},
    {1, 2},
    {0, 1},
    {2, 3},
    {2, 3}
]

print(mcla(comm, 2))

