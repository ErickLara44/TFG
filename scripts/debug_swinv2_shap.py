import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_prop_improved import DEFAULT_FEATURE_VARS as CHANNELS

FEATURE_VARS = list(CHANNELS) + ['Fire_Mask_T0']

# Load the saved SHAP values or we just need to re-run the explanation on 1 sample.
# Since it takes a minute, let's just re-run explain_swinv2_shap.py but add a print statement to the existing file.
