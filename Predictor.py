import pandas as pd


class Predictor(object):

    def _load_from_file(self, file_path):
        keywords_set = set()
        f_keywords = open(file_path, "r")
        keywords_lines = [line.rstrip('\r\n').split(',') for line in f_keywords.readlines()]
        f_keywords.close()
        for ak_doc in keywords_lines:
            keywords_set.update(ak_doc)
        return keywords_lines, keywords_set

    def _compute_occurrence_matrix(self, ak_lines, ak_set, kp_lines, kp_set):
        self._occurrence_matrix = pd.DataFrame(0, index=kp_set, columns=ak_set, dtype=int)

        for ak_doc, kp_doc in zip(ak_lines, kp_lines):
            for kp_word in kp_doc:
                for ak_word in ak_doc:
                    self._occurrence_matrix[ak_word][kp_word] += 1

    def _compute_frequency_matrix(self):
        self._frequency_matrix = self._occurrence_matrix.div(self._occurrence_matrix.sum(axis=1), axis=0)

    def generate_from_file(self, ak_file_path, kp_file_path):
        ak_lines, ak_set = self._load_from_file(ak_file_path)
        kp_lines, kp_set = self._load_from_file(kp_file_path)

        self._compute_occurrence_matrix(ak_lines, ak_set, kp_lines, kp_set)
        self._compute_frequency_matrix()

    def compute_kp_probabilities(self, ak_set):
        filtered_frequency_matrix = self._frequency_matrix[ak_set]
        kp_probabilites = filtered_frequency_matrix.sum(axis=1)
        return kp_probabilites.sort_values(ascending=False).to_frame(name="frequency")

    def compute_kp(self, ak_set, energy = 0.5):
        kp_probabilities = self.compute_kp_probabilities(ak_set)
        total_values = kp_probabilities.sum()
        energy_threshold = float(total_values * energy)
        kp_probabilities['cumulative_frequency'] = kp_probabilities.cumsum()
        kp_probabilities_series = kp_probabilities[kp_probabilities['cumulative_frequency'] >= energy_threshold ]['frequency']
        return kp_probabilities_series.to_dict()


    def compute_kp_set(self, ak_set, energy = 0.5):
        return set(self.compute_kp(ak_set, energy).keys())
