import numpy as np
from sample import sample, sample_multi
from scipy.spatial.distance import cosine

DEFAULT_MODEL = "./save/sentiment/lstm_cnn"

def get_vectors():
    pass

def read_data():
    pass


def get_vector_by(word, path_to_model=DEFAULT_MODEL):
    return sample(path_to_model, word)


# hoho = get_vector_by("woman king queen man")
# woman = hoho[0]
# king = hoho[1]
# queen = hoho[2]
# man = hoho[3]
# woman_man = 1 - cosine(woman, man)
# king_queen = 1 - cosine(king, queen)
#
# man_king = 1 - cosine(man, king)
# woman_queen = 1 - cosine(woman, queen)
#
# print("Distances")
# print("woman_man:{} \n king_queen:{} \n".format(woman_man, king_queen))
# print("man_king:{} \n woman_queen:{} \n".format(man_king, woman_queen))
#
# print("Queen similarity")
# print(1 - cosine(king - man + woman, queen))


pos1 = "You have no affinity for most of the characters ."
pos2 = "The characters , cast in impossibly contrived situations , are totally estranged from reality ."

neg1 = "Everybody loves a David and Goliath story , and this one is told almost entirely from David 's point of view ."
neg2 = "Those who want to be jolted out of their gourd should drop everything and run to Ichi."
positive = [pos1,pos2]
negative = [neg1, neg2]
vec =  positive+negative
print(len(vec))
results = sample_multi(DEFAULT_MODEL,vec, "biLSTM")
pos_pos = 1 - cosine(results[0], results[1])
neg_neg = 1 - cosine(results[2], results[3])
pos_neg = 1 - cosine(results[1], results[3])
neg_pos = 1 - cosine(results[2], results[0])

print("pos_pos {}".format(pos_pos))
print("neg {}".format(neg_neg))
print("neg_pos {}".format(neg_pos))
print("pos_neg {}".format(pos_neg))