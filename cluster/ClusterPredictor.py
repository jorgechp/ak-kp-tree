import numpy as np

from ordered_set import OrderedSet

from AbstractPredictor import AbstractPredictor
from sklearn.feature_extraction.text import CountVectorizer


class ClusterPredictor(AbstractPredictor):

    def _load_from_file(self, file_path, split=True):
        f_keywords = open(file_path, "r")
        if split:
            keywords_lines = [line.rstrip('\r\n').split(',') for line in f_keywords.readlines()]
        else:
            keywords_lines = [line.rstrip('\r\n') for line in f_keywords.readlines()]
        f_keywords.close()
        return keywords_lines

    def _generate_set_from_lines(self, lines):
        keyword_set = set()

        for keyword in lines:
            keyword_set.update(keyword)
        return keyword_set

    def _jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def generate_from_file(self, ak_file_path, kp_file_path):
        ak_lines = self._load_from_file(ak_file_path, split=False)
        self._kp_clusters = self._load_from_file(kp_file_path, split=True)

        vectorizer = CountVectorizer()

        self._ak_matrix = vectorizer.fit_transform(ak_lines)
        del ak_lines
        self._ak_keywords = vectorizer.get_feature_names()

    def compute_kp_scores(self, ak_set):
        ak_indices = [self._ak_keywords.index(keyword) for keyword in ak_set]

        sequencies_indices = OrderedSet()
        list_append_call = sequencies_indices.update
        for ak_index in ak_indices:
            result_matrix = self._ak_matrix[:, ak_index].nonzero()[0]
            seq_index = [res for res in result_matrix]
            list_append_call(seq_index)

        sequence_list = []
        for sequency_index in sequencies_indices:
            complete_sequence_index = self._ak_matrix[sequency_index].indices
            sequence_list.append([self._ak_keywords[index] for index in complete_sequence_index])

        jaccard_distance_call = self._jaccard_similarity
        scores = np.array([jaccard_distance_call(ak_set, sequence) for sequence in sequence_list])

        argmax = np.argmax(scores)
        selected_cluster = sequencies_indices[argmax]
        return self._kp_clusters[selected_cluster]









