import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
def data_processing(input_files):
    output_data = []
    for file in input_files:
        x = list()
        y = list()
        with open(file) as f:
            array = json.load(f)
            data = array['content']
            X = np.empty(shape=(0, 2))
            np.asarray(X)
            for translation in data:
                for lang, info in translation.items():
                    x.append(info[1][0])
                    y.append(info[1][1])
                    new = np.array(info[1])
                    X = np.append(X, new.reshape((-1, 2)), axis=0)
        output_data.append(X)
    return output_data


# #############################################################################
# Compute DBSCAN

data_db = data_processing(['/Users/lolo/Documents/visualization/decodings_es_layer0.json',
                           '/Users/lolo/Documents/visualization/decodings_es_layer1.json',
                           '/Users/lolo/Documents/visualization/decodings_es_layer2.json',
                           '/Users/lolo/Documents/visualization/decodings_es_layer3.json',
                           '/Users/lolo/Documents/visualization/decodings_es_layer4.json',
                           '/Users/lolo/Documents/visualization/decodings_es_layer5.json'])

for ind, X in enumerate(data_db):  # TODO export these clusters in the final visualization
    db = DBSCAN(eps=0.5, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # #############################################################################
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        plt.figure(ind)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
