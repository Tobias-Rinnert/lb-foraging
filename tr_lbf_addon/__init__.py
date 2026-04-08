import sys
import os

# Ensure bare imports (lbf_elements, lbf_gym, etc.) resolve correctly
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

from .lbf_gym import LBF_GYM
