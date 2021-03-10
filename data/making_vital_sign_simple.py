import numpy as np
import pandas as pd
import pickle

# few data: DBPABPVALUE, SBPABPVALUE, RRRRVALUE

list_ids = pd.read_csv('tidy_Stroke_Vital_Sign.csv')[['UID', 'Hospital_ID']]

# version 1
bt_org = pd.read_csv('BTVALUE.csv')
bt = pd.merge(list_ids, bt_org, how='left', on=['UID', 'Hospital_ID']).drop(['UID', 'Hospital_ID', 'TYPE', 'SUBCLASS'], axis=1)
bt['sep'] = -1

dbp_org = pd.read_csv('DBPNBPVALUE.csv')
dbp = pd.merge(list_ids, dbp_org, how='left', on=['UID', 'Hospital_ID']).drop(['UID', 'Hospital_ID', 'TYPE', 'SUBCLASS'], axis=1)
dbp['sep'] = -1

hr_org = pd.read_csv('HRHRVALUE.csv')
hr = pd.merge(list_ids, hr_org, how='left', on=['UID', 'Hospital_ID']).drop(['UID', 'Hospital_ID', 'TYPE', 'SUBCLASS'], axis=1)
hr['sep'] = -1

pu_org = pd.read_csv('HRPULSVALUE.csv')
pu = pd.merge(list_ids, pu_org, how='left', on=['UID', 'Hospital_ID']).drop(['UID', 'Hospital_ID', 'TYPE', 'SUBCLASS'], axis=1)
pu['sep'] = -1

sbp_org = pd.read_csv('SBPNBPVALUE.csv')
sbp = pd.merge(list_ids, sbp_org, how='left', on=['UID', 'Hospital_ID']).drop(['UID', 'Hospital_ID', 'TYPE', 'SUBCLASS'], axis=1)
sbp['sep'] = -1

temp = pd.concat([bt, dbp, hr, pu, sbp], axis=1).values
final = []
inx = 0
for r in temp:
    cleanedList = r[~np.isnan(r)]
    final.append(cleanedList)
list_ids['vital_signs'] = final

with open('vital_sign_simple.pickle', 'wb') as f:
    pickle.dump(list_ids, f)
# list_ids.to_csv('vital_sign_simple.csv', index=False)
print('done')