"""
Here I want to test where the query of the secrets happen if I have it coded in the momics module

It works, no .env file is needed in the momics package, the .env file is needed in the project folder
"""


import os
import sys
import requests
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import init_setup
init_setup()

# All low level functions are imported from the momics package
from momics.loader import get_ro_crate_metadata_gh

sample_id = "EMOBON_BPNS_So_34"
metadata_json = get_ro_crate_metadata_gh(sample_id)
print(metadata_json)



