# TODO

import csv
import logging
from pprint import pprint

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from sound_monitor.config import Config

_config = Config.get()
_logger = logging.getLogger(__name__)

interpreter = Interpreter(model_path=str(_config.yamnet_model))

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]["index"]
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]["index"]

length = 0.96 # seconds
waveform = np.zeros(int(length * 16000), dtype=np.float32)
# - 0.96 second window
# - 0.48 second hop
# - last window might be padded
# ---
# - 1s gives two windows
# - 0.96s gives one window

interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
interpreter.allocate_tensors()
interpreter.set_tensor(waveform_input_index, waveform)
interpreter.invoke()

scores: np.ndarray[np.float32] = interpreter.get_tensor(scores_output_index)
print(scores.shape)


def class_names_from_csv(csv_file_path):
    """Returns list of class names from a CSV file."""
    with open(csv_file_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [row[2] for row in reader]


class_names = class_names_from_csv(_config.yamnet_class_map)
print(class_names[scores.mean(axis=0).argmax()])  # Should print 'Silence'.

means = scores.mean(axis=0)
pairs = [(x, y) for x, y in sorted(zip(means, class_names)) if x > 0]
pprint(pairs)
