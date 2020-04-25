import os
import json

import requests
from tqdm import tqdm

if "LEADERS_PRIZE_API_KEY" not in os.environ:
    raise ValueError("LEADERS_PRIZE_API_KEY not found in environemnt variables.")
