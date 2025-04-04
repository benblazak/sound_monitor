try:
    # raspberry pi
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # mac
    from tensorflow.lite.python.interpreter import Interpreter
