import os
import pickle

from MeanProbabilityPredictor import MeanProbabilityPredictor
from ProbabilityPredictor import ProbabilityPredictor
from Validator import Validator
from cluster.ClusterPredictor import ClusterPredictor
from nn import NN_predictor
from nn.NN_predictor import NNPredictor

AK_FILE = "data/ak_5325.txt"
KP_FILE = "data/kp_5325.txt"
PREDICTOR_MODEL_PATH = "saved_models/predictor_probability_5325_training09_energy08_akkp.pkl"
NN_DATA_MODEL_PATH = "saved_models/data_model_5325_ak_kp.pt"
NN_DATA_MANAGER_PATH = "saved_models/data_manager_5325_ak_kp.pkl"


TRAINING_RATE = 0.9
ENERGY_RATE = 0.99
ENERGY_CLUSTER= 0.3



if os.path.isfile(PREDICTOR_MODEL_PATH):
    with open(PREDICTOR_MODEL_PATH, 'rb') as pickle_file:
        predictor = pickle.load(pickle_file)
else:
    predictor = ProbabilityPredictor()
    predictor.generate_from_file(AK_FILE, KP_FILE)
    pickle.dump(predictor, open(PREDICTOR_MODEL_PATH, "wb"), protocol=4)




print("Validation 1: Probabilistic")
validator = Validator()
validator.load_validator(AK_FILE, KP_FILE)
validator.split_training_test(training_rate=TRAINING_RATE)
results = validator.validate(predictor, energy=ENERGY_RATE)
print(results)

del predictor
del validator

print("Validation 2: Cluster")
predictor_cluster = ClusterPredictor()
predictor_cluster.generate_from_file(AK_FILE, KP_FILE)

validator_cluster = Validator()
validator_cluster.load_validator(AK_FILE, KP_FILE)
validator_cluster.split_training_test(training_rate=TRAINING_RATE)
results = validator_cluster.validate(predictor_cluster, energy=ENERGY_CLUSTER)
print(results)

del predictor_cluster
del validator_cluster

print("Validation 3: Neural network")
nn_predictor = NNPredictor()
nn_predictor.prepare(NN_DATA_MODEL_PATH, NN_DATA_MANAGER_PATH)


validator_nn = Validator()
validator_nn.load_validator(AK_FILE, KP_FILE)
validator_nn.split_training_test(training_rate=TRAINING_RATE)
results = validator_nn.validate(nn_predictor, energy=ENERGY_RATE)
print(results)

del nn_predictor
del validator_nn


# print("Validation 2")
#
# validator = Validator()
# validator.load_validator(AK_FILE, KP_FILE)
# validator.split_training_test(training_rate=TRAINING_RATE)
# results = validator.validate(MeanProbabilityPredictor, energy=ENERGY_RATE)
# print(results)






