"""
Filter shipment data down to 181 samplings with real sequencing data for batch 1 and 2.

What gets discarded is the following:
- All samples which did not get sequenced fro whatever reason
- MOCKS
- BLANKS

The result is the 181 samples are sequenced and kept, 39 get discarded from the shipment files
"""

import os
import sys
# add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd


BATCH1_RUN_INFO_PATH = (
    "https://raw.githubusercontent.com/emo-bon/sequencing-data/main/shipment/"
    "batch-001/run-information-batch-001.csv"
)
BATCH2_RUN_INFO_PATH = (
    "https://raw.githubusercontent.com/emo-bon/sequencing-data/main/shipment/"
    "batch-002/run-information-batch-002.csv"
)


## method to merge tracker and shipments data
def merge_tracker_metadata(df_tracker):
    df = pd.read_csv(BATCH1_RUN_INFO_PATH)
    df1 = pd.read_csv(BATCH2_RUN_INFO_PATH)
    df = pd.concat([df, df1])
    print(len(df))

    # select only batch 1 and 2 from tracker data
    df_tracker = df_tracker[df_tracker['batch'].isin(['001', '002'])]

    # merge with tracker data on ref_code
    df = pd.merge(df, df_tracker, on='ref_code', how='inner')

    # split the sequenced tables
    df_discarded = df[df['lib_reads_seqd'].isnull()]
    df1 = df[~df['lib_reads_seqd'].isnull()]

    # deliver only filters and sediment samples, no blanks
    df_to_deliver = df1[df1['sample_type'].isin(['filters', 'sediment'])].reset_index(drop=True)
    #drop columns
    df_to_deliver.drop(columns='reads_name_y', inplace=True)
    df_to_deliver.rename(columns={'reads_name_x': 'reads_name'}, inplace=True)
    return df_to_deliver, df_discarded


if __name__ == "__main__":
    # define root directory which is where the file is located
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Root directory: {root_dir}")


    path_tracker = os.path.join(root_dir, '../wf0_landing_page/emobon_sequencing_master.csv')
    # load tracker data
    df_tracker = pd.read_csv(path_tracker ,index_col=0)
    # merge with run information
    df_to_deliver, df_discarded = merge_tracker_metadata(df_tracker)

    # save to file
    save_dir = os.path.join(root_dir, '../data')
    os.makedirs(save_dir, exist_ok=True)
    df_to_deliver.to_csv(
        os.path.join(root_dir, '../data/shipment_b1b2_181.csv'),
        index=False,
        )
    df_discarded.to_csv(
        os.path.join(root_dir, '../data/shipment_b1b2_39_discarded.csv'),
        index=False,
        )
