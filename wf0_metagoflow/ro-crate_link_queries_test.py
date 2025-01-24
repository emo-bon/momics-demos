"""
BEWARE: For private repos, you need to use the gihub API and authorization token and then also adjust the URL
by using `api.githu.com` and also putting `contents` in the URL path.
"""


import os
import sys
import requests
import json
from dotenv import load_dotenv
load_dotenv()


sample_id = "EMOBON_BPNS_So_34"
username = os.getenv('GH_USER')
token = os.getenv('GH_TOKEN')

# login example #
#################
# login = requests.get('https://api.github.com/search/repositories?q=github+api',
#                      auth=(username,token))
# print('login status', login.status_code)


# Private repo example #
########################
# Not the future of ro-crates retrieval, but a current reality
metdat = f"https://api.github.com/repos/emo-bon/metaGOflow-rocrates-dvc/contents/{sample_id}-ro-crate/ro-crate-metadata.json"
private_req = requests.get(metdat, 
                  headers={
                    'accept': 'application/vnd.github.v3.raw',
                    'authorization': f'token {token}',
                    })
print('ro-crate-metadata.json request 1', private_req.status_code)

# print response text
print(private_req.json())


# Example for PUBLIC repos, no privacy #
########################################
# public_json = "https://github.com/palec87/ImSwitchOpt/blob/master/imswitch/_data/user_defaults/imcontrol_slm/info_oil.json"
# rpublic = requests.get(public_json, headers={'accept': 'application/json'})
# print('info_oil.json request', rpublic.status_code)
# print(rpublic.json())