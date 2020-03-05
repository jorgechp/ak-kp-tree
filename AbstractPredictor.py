from abc import ABC, abstractmethod

class AbstractPredictor(ABC):
    def _load_from_file(self, file_path):
        f_keywords = open(file_path, "r")
        keywords_lines = [line.rstrip('\r\n').split(',') for line in f_keywords.readlines()]
        f_keywords.close()
        return keywords_lines

    def _generate_set_from_lines(self, lines):
        keyword_set = set()

        for keyword in lines:
            keyword_set.update(keyword)
        return keyword_set

    def generate_from_file(self, ak_file_path, kp_file_path):
        ak_lines = self._load_from_file(ak_file_path)
        kp_lines = self._load_from_file(kp_file_path)

        ak_set = self._generate_set_from_lines(ak_lines)
        kp_set = self._generate_set_from_lines(kp_lines)

        self._compute_occurrence_matrix(ak_lines, ak_set, kp_lines, kp_set)
        self._compute_frequency_matrix()

    def generate_from_lines(self, ak_lines, kp_lines):
        ak_set = self._generate_set_from_lines(ak_lines)
        kp_set = self._generate_set_from_lines(kp_lines)

        self._compute_occurrence_matrix(ak_lines, ak_set, kp_lines, kp_set)
        self._compute_frequency_matrix()

    def compute_kp(self, ak_set, energy = 0.7):
        kp_scores = self.compute_kp_scores(ak_set).sort_values(ascending=False).to_frame(name="score")
        total_values = kp_scores.sum()
        energy_threshold = float(total_values * energy)
        kp_scores['cumulative_score'] = kp_scores.cumsum()
        kp_scores_series = kp_scores[kp_scores['cumulative_score'] >= energy_threshold]['score']
        return kp_scores_series.to_dict()

    def compute_kp_set(self, ak_set, energy = 0.7):
        return set(self.compute_kp(ak_set, energy).keys())

    @abstractmethod
    def _compute_occurrence_matrix(self, ak_lines, ak_set, kp_lines, kp_set):
        pass

    @abstractmethod
    def _compute_frequency_matrix(self):
        pass

    @abstractmethod
    def compute_kp_scores(self, ak_set):
        pass
