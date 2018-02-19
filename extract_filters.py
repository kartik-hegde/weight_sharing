import numpy as np

data = np.load("data_kmeans.npy").item()

conv3 = np.asarray(data['conv3'][0], dtype=float)

R = len(conv3)
S = len(conv3[0])
C = len(conv3[0][0])
K = len(conv3[0][0][0])

filters = []

for k in range(K):
    filters.append([])
    for c in range(C):
        filters[k].append([])
        for r in range(R):
            filters[k][c].append([])
            for s in range(S):
                filters[k][c][r].append(conv3[r][s][c][k])


np.save("data_filter.npy", np.asarray(filters))
