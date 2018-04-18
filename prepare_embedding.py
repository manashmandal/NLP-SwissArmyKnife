import numpy as np

error_words = []
embedding_matrix = np.zeros((NUM_TOKENS + 1, EMBEDDING_DIM))
error = 0
for idx, word in tqdm(enumerate(vocabulary)):
    try:
        embedding_matrix[idx + 1] = glove_embeddings[vocabulary[idx]]
    except:
        error_words.append(vocabulary[idx])
        error +=1
np.save('embedding_matrix.npy', embedding_matrix)
