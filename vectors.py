import numpy as np
from sample import sample
from scipy.spatial.distance import cosine

DEFAULT_MODEL = "./save/MRPC/stacked_bilstm"

def get_vectors():
    pass

def read_data():
    pass


def get_vector_by(word, path_to_model=DEFAULT_MODEL):
    return sample(path_to_model, word)

hoho = get_vector_by("woman king queen man")
woman = hoho[0]
king = hoho[1]
queen = hoho[2]
man = hoho[3]
woman_man = 1 - cosine(woman, man)
king_queen = 1 - cosine(king, queen)

man_king = 1 - cosine(man, king)
woman_queen = 1 - cosine(woman, queen)

print("Distances")
print("woman_man:{} \n king_queen:{} \n".format(woman_man, king_queen))
print("man_king:{} \n woman_queen:{} \n".format(man_king, woman_queen))

print("Queen similarity")
print(1 - cosine(king - man + woman, queen))