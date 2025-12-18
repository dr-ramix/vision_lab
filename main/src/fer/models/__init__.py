from .factory import build_model, register_model, available_models

# IMPORTANT:
# importing model modules here ensures they register themselves
from . import cnn_simple  # noqa: F401