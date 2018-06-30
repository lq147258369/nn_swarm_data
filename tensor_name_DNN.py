import os
from tensorflow.python import pywrap_tensorflow
import numpy as np
np.set_printoptions(threshold=np.inf)

checkpoint_path = os.path.join( "F:\Bristol\Robotics\Dissertation\code\9robots_model\model.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))