from gensim.models import Word2Vec
import pickle

w2v_model_file = "models/word2vec.model"
w2v_model_dict = "data/processed/word2vec_dict.pkl"

print("Loading Word2Vec model ")
model = Word2Vec.load(w2v_model_file)

print(" Creating dictionary of word2Vec")
word2vec_dict = {token: model.wv[token] for token in model.wv.index_to_key}

print(f"Total words in vocabulary: {len(word2vec_dict)}")
with open(w2v_model_dict, "wb") as f:
    pickle.dump(word2vec_dict, f)

print(f" Word2Vec dictionary is saved at {w2v_model_dict}")
