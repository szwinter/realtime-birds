#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 16:26:07 2025

@author: gtikhono
"""

import boto3
import os
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
disable_inner_tqdm = True

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(2**31), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

try:
    path_data = os.environ["RTB_SCRATCH"]
except Exception:
    print("RTB_SCRATCH environmental variable is not set, use local directory")
    path_data = "."

os.makedirs(os.path.join(path_data, "allas_import", "json"), exist_ok=True)
os.makedirs(os.path.join(path_data, "allas_import", "metadata"), exist_ok=True)
os.makedirs(os.path.join(path_data, "allas_import", "observations"), exist_ok=True)


s3_resource = boto3.resource('s3', endpoint_url='https://a3s.fi')
json_bucket = s3_resource.Bucket('2007581-mongodb-data-json')
object_list = list(json_bucket.objects.all())
df_columns = ['user', 'date', 'time', 'len', 'dur', 'real_obs', 'rec_type', 'point_count_loc', 'lat', 'lon', 'rec_id']
df_rec_colnames = ['species', 'prediction', 'orig_prediction', 'rec_id', 'result_id']

for k, obj in enumerate(tqdm(object_list)):
  path = os.path.join(path_data, "allas_import", "json", obj.key)
  if not os.path.exists(path):
    obj.Object().download_file(path)
  else:
    md5_s3 = obj.e_tag[1:-1]
    md5_local = md5(path)
    if md5_s3 != md5_local:
      obj.Object().download_file(path)
    # print(k, md5_s3, md5_local)
  
  with open(path) as json_file:
    json_data = json.load(json_file)
  df = pd.DataFrame(columns=df_columns, index=np.arange(len(json_data)))
  rec_df_list = []
  for i, elem  in enumerate(tqdm(json_data, leave=False, disable=disable_inner_tqdm)):
    df.loc[i, 'user'] = elem['shortkey']
    if 'length' in elem['sampleAudioFile'].keys():
      df.loc[i, 'len'] = elem['sampleAudioFile']['length']    
    # if 'audioFileURL' in d[i]['sampleAudioFile'].keys():
    df.loc[i, 'rec_id'] = elem['_id']
    
    if 'sampleMetadata' in elem.keys(): # v1
      smd_key = 'sampleMetadata'
    elif 'sampleMetadatav2' in elem.keys(): # v2
      smd_key = 'sampleMetadatav2'
    elif 'sampleMetadatav3' in elem.keys(): # v3
      smd_key = 'sampleMetadatav3'
    else:
      print(f"Sample metadata not found for index {i} in date {obj.key}")
    smd = elem[smd_key]
    
    if smd_key in ['sampleMetadatav2', 'sampleMetadatav3']: # save rec_type if metadata at least v2
      rec_type = smd['sample_type']
      df.loc[i, 'rec_type'] = rec_type
      if rec_type == "point": # pistelaskentapisteen nimi
        if ('target_name' in smd['point'].keys()) & ('point_name' in smd['point'].keys()):
          df.loc[i, 'point_count_loc'] = smd['point']['target_name'] + '_' + smd['point']['point_name']
    
    df.loc[i, ['date', 'time']] = smd['datetime'].split(' ')
    df.loc[i, 'real_obs'] = smd['realobservation']
    df.loc[i, ['lat','lon']] = smd['location']['lat'], smd['location']['long']
    if smd['realobservation'] == True:
      if 'resultexist' in elem.keys() and elem['resultexist'] and type(elem['result']) == list:
        df_rec = pd.DataFrame(columns=df_rec_colnames, index=np.arange(len(elem['result'])))
        for j, res in enumerate(elem['result']):
          if smd_key == 'sampleMetadatav3': # v3
            df_rec.loc[j, "species"] = res["birdname"]["iczn"]
          else: #v1, v2
            df_rec.loc[j, "species"] = res["scientificname"]
          df_rec.loc[j, "prediction"] = res["probability"]
          df_rec.loc[j, "result_id"] = res["resultuuid"]
          df_rec.loc[j, "rec_id"] = elem['_id']
          if 'original_probability' in res.keys():
            df_rec.loc[j, "original_probability"] = res['original_probability']
        rec_df_list.append(df_rec)
    
  
  df['dur'] = df['len'] / 8200
  df.dropna(subset="len", inplace=True)
  df = df.loc[df['real_obs']==True]
  if len(rec_df_list) > 0:
    rec_df_combined = pd.concat(rec_df_list)
  else:
    rec_df_combined = pd.DataFrame(columns=df_rec_colnames, index=np.arange(0))
  
  df.to_csv(os.path.join(path_data, "allas_import", "metadata", f"{k:05}_{obj.key.split('.')[0]}.csv"), index=False)
  rec_df_combined.to_csv(os.path.join(path_data, "allas_import", "observations", f"{k:05}_{obj.key.split('.')[0]}.csv"), index=False)
    

  
# s3_resource.Object('2007581-mongodb-data-json', 'db_data_2025_5_20.json').download_file('local_file.json')

