import os
import pickle

from MeanProbabilityPredictor import MeanProbabilityPredictor
from ProbabilityPredictor import ProbabilityPredictor
from Validator import Validator

AK_FILE = "full_dataset_ak.txt"
KP_FILE = "full_dataset_kp.txt"
PREDICTOR_MODEL_PATH = "predictor_probability.pkl"

# predictor = ProbabilityPredictor()
# predictor.generate_from_file(AK_FILE, KP_FILE)
# print(predictor.compute_kp_scores({'technicalanalysis','sentimentembeddings','markettrendprediction','supervisedlearning','timeseriesanalysis','sentimentanalysis'}))
# # predictor.compute_kp_scores({'petalo'})
# # predictor.compute_kp_scores({'concurrency'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills','petalo'})
# # predictor.compute_kp_scores({'petalo', 'concurrency'})
#
#
# print("Mean predictor")
#
# predictor = MeanProbabilityPredictor()
# predictor.generate_from_file(AK_FILE, KP_FILE)
# # print(predictor.compute_kp_scores({'technicalanalysis','sentimentembeddings','markettrendprediction','supervisedlearning','timeseriesanalysis','sentimentanalysis'}))
# # predictor.compute_kp_scores({'concurrency'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills'})
# # predictor.compute_kp_scores({'concurrency', 'parallelprogramming','activelearning','softwareengineering','softskills','petalo'})
# # predictor.compute_kp_scores({'petalo', 'concurrency'})

TRAINING_RATE = 0.9
ENERGY_RATE = 1.0


if os.path.isfile(PREDICTOR_MODEL_PATH):
    with open(PREDICTOR_MODEL_PATH, 'rb') as pickle_file:
        predictor = pickle.load(pickle_file)
else:
    predictor = ProbabilityPredictor()
    predictor.generate_from_file(AK_FILE, KP_FILE)
    pickle.dump(predictor, open(PREDICTOR_MODEL_PATH, "wb"), protocol=4)


print("Validation 1")
validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=TRAINING_RATE)
results = validator.validate(predictor, energy=ENERGY_RATE)
print(results)

# print("Validation 2")
#
# validator = Validator()
# validator.load_validator(AK_FILE, KP_FILE)
# validator.split_training_test(training_rate=TRAINING_RATE)
# results = validator.validate(MeanProbabilityPredictor, energy=ENERGY_RATE)
# print(results)






