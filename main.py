import os
import pickle

from MeanProbabilityPredictor import MeanProbabilityPredictor
from ProbabilityPredictor import ProbabilityPredictor
from Validator import Validator
from nn import NN_predictor
from nn.NN_predictor import NNPredictor

AK_FILE = "data/ak_5325.txt"
KP_FILE = "data/ak_5325.txt"
PREDICTOR_MODEL_PATH = "saved_models/predictor_probability_5325_training09_energy08.pkl"
NN_DATA_MODEL_PATH = "saved_models/data_model.pt"
NN_DATA_MANAGER_PATH = "saved_models/data_manager.pkl"


TRAINING_RATE = 0.9
ENERGY_RATE = 0.8


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

print("Neural network")
nn_predictor = NNPredictor()
nn_predictor.prepare(NN_DATA_MODEL_PATH, NN_DATA_MANAGER_PATH)


validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=TRAINING_RATE)
results = validator.validate(nn_predictor, energy=ENERGY_RATE)
print(results)




# print("Validation 2")
#
# validator = Validator()
# validator.load_validator(AK_FILE, KP_FILE)
# validator.split_training_test(training_rate=TRAINING_RATE)
# results = validator.validate(MeanProbabilityPredictor, energy=ENERGY_RATE)
# print(results)






