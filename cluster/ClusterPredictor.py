import re

import numpy as np
import pandas as pd

from ordered_set import OrderedSet

from AbstractPredictor import AbstractPredictor
from sklearn.feature_extraction.text import CountVectorizer


class ClusterPredictor(AbstractPredictor):


    def _load_from_file(self, file_path, split=True):
        f_keywords = open(file_path, "r")
        if split:
            keywords_lines = [self._clean(line).rstrip('\r\n').split(',') for line in f_keywords.readlines()]
        else:
            keywords_lines = [self._clean(line).rstrip('\r\n') for line in f_keywords.readlines()]
        f_keywords.close()
        return keywords_lines

    def _generate_set_from_lines(self, lines):
        keyword_set = set()

        for keywords in lines:
            keyword_set.update(keywords.split(","))
        return keyword_set

    def _jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def _process_scores(self, sequence_list, ak_set, sequencies_indices):
        jaccard_distance_call = self._jaccard_similarity
        scores = np.array([jaccard_distance_call(ak_set, sequence) for sequence in sequence_list])
        scores_dict = {}

        for index, cluster in enumerate(scores):
            cluster_score = scores[index]
            selected_cluster = sequencies_indices[index]
            cluster = self._kp_clusters[selected_cluster]

            for keyword in cluster:
                scores_dict[keyword] = cluster_score

        return pd.Series(scores_dict)

    def compute_existing_keywords(self, ak_set):
        return self._kp_set.intersection(ak_set)

    def generate_from_file(self, ak_file_path, kp_file_path):
        ak_lines = self._load_from_file(ak_file_path, split=False)
        self._kp_clusters = self._load_from_file(kp_file_path, split=True)
        self._kp_set = self._generate_set_from_lines(ak_lines)

        vectorizer = CountVectorizer(strip_accents=None,token_pattern = r"(?u)\b\w+\b")

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

        return self._process_scores(sequence_list, ak_set, sequencies_indices)

    def compute_kp(self, ak_set, energy=0.7):
        existing_keywords = self.compute_existing_keywords(ak_set)

        kp_scores = self.compute_kp_scores(existing_keywords)


        if kp_scores is not False:
            kp_scores = kp_scores.sort_values(ascending=False).to_frame(name="score")
            kp_scores = kp_scores[kp_scores['score'] >= energy]['score']
            return kp_scores.to_dict()
        else:
            return dict()











