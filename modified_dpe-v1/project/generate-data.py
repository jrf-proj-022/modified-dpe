# essential imports
from model.cy_utils import ar1_data_generator, skewtent_data_generator, extract_json
import json

# AR(1) synthetic unidirectional coupling
ar1_data_generator(num_sequences = 2000, sequence_length = 2000, transients =500)

# # verifying generated data
# ar_filepath = 'dataset/ar1_data.json'
# ar_data = extract_json(ar_filepath)
# print('Keys: ', ar_data.keys())
# print('Sample: ', ar_data['0.0']['X'][0][:100])

# 1D Coupled Sketent-maps
# skewtent_data_generator(b1=0.35, b2=0.76, initial_values=None)

# verifying generated data
skewtent_filepath = 'dataset/skewtent_maps_data.json'
skewtent_data = extract_json(skewtent_filepath)

print('Keys: ', skewtent_data.keys())
print('Sample: ', skewtent_data['0.0']['X'][0][:100])
