#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Quan Yuan
"""
import os
import pandas as pd
from google.cloud import bigquery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "googlemap-4dc7a6055579 5.25.37 PM.json"
client = bigquery.Client()

def read_data(query):
    df = pd.io.gbq.read_gbq(query, dialect='standard')
    return df

