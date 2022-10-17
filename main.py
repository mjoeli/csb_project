import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import os
from scipy.stats import ttest_ind

rel_data_dir_full = './data/sharew8_all/'
rel_data_dir_easy = './data/sharew8_easy/'
plot_dir = './plots/'
outdir = './'

'''
Wave 8 Release 8.0.0
DOI: 10.6103/SHARE.w8.800
2022-02-10 00:53:53
'''
all_files = glob(f"{rel_data_dir_full}*.dta")

'''
easySHARE Release 8.0.0
DOI: 10.6103/SHARE.easy.800
2022-02-10 01:00:22
'''
easy_files = glob(f"{rel_data_dir_easy}*.dta")

'''
Wave 8 Accelerometer Epochs Release 8.0.0
DOI: 10.6103/SHARE.w8.800
2022-02-10 01:04:04
'''
acc_files = glob(f"{rel_data_dir_full}dta/**/*.dta", recursive=True)

df_easy = pd.read_stata(easy_files[0])
df_ph = pd.read_stata(os.path.join(rel_data_dir_full, 'sharew8_rel8-0-0_ph.dta'))
df_act = pd.read_stata(os.path.join(rel_data_dir_full, 'sharew8_rel8-0-0_br.dta'))
df_acc_hr = pd.read_stata(os.path.join(rel_data_dir_full, 'sharew8_rel8-0-0_gv_accelerometer_hour.dta'))
df_acc = pd.read_stata(os.path.join(rel_data_dir_full, 'sharew8_rel8-0-0_gv_accelerometer_total.dta'))
df_acc_imp = pd.read_stata(os.path.join(rel_data_dir_full, 'sharew8_rel8-0-0_gv_imputations.dta'))
df_acc_hr_clean = df_acc_hr[df_acc_hr['GGIR_N_valid_hours'] > 0.0].iloc[:, 0:15]


# function to merge df1 and df2 by ID
def get_common_columns(df1, df2, merge_on=['mergeid']):
    common = df1.columns.difference(df2.columns).tolist()
    common.extend(merge_on)
    return common


df = df_acc_hr_clean.copy()
df = pd.merge(left=df, left_on=['mergeid'], right=df_ph[get_common_columns(df_ph, df)], right_on=['mergeid'],
              how='left')
df = pd.merge(left=df, left_on=['mergeid'], right=df_act[get_common_columns(df_act, df)], right_on=['mergeid'],
              how='left')
df = pd.merge(left=df, left_on=['mergeid'], right=df_easy[get_common_columns(df_easy, df)], right_on=['mergeid'],
              how='left')
df = df[df['wave'] == 8]


def map_selected_num(df, columns: list[str]):
    df = df.copy()
    for c in columns:
        df.loc[:, c] = df[c].map({
            'Selected': 1,
            'Not selected': 0,
            "Don't know": -1,
            'Refusal': -2,
            1: 1,
            0: 0,
            -1: -1,
            -2: -2,
        })
    return df

#                        Hypertension, Diabetis, Osteoarthritis
df = map_selected_num(df, ['ph006d2', 'ph006d5', 'ph006d20'])


def map_bmi_num(df, columns: list[str] = ['bmi']):
    df = df.copy()
    for c in columns:
        df.loc[:, c] = df[c].replace({
            '-15. no information': -1,
            '-13. not asked in this wave': -1,
            "-12. don't know / refusal": -1,
            '-3. implausible value/suspected wrong': -1,
        })
        df[c] = df[c].astype(float)
    return df


df = map_bmi_num(df)


def map_freq_num(df, columns: list[str]):
    df = df.copy()
    for c in columns:
        df.loc[:, c] = df[c].replace({
            'More than once a week': 1,
            'Once a week': 2,
            'One to three times a month': 3,
            'Hardly ever, or never': 4,
            "Don't know": -1,
            'Refusal': -1,
        })
        df[c] = df[c].astype(float)
    return df


df = map_freq_num(df, ['br015_', 'br016_'])


def calc_ENMO_stats(df, enmo_col='GGIR_mean_ENMO_hour'):
    df = df.copy()
    mean = df.groupby('mergeid')[enmo_col].mean()
    median = df.groupby('mergeid')[enmo_col].median()
    var = df.groupby('mergeid')[enmo_col].var()
    mean.name = 'mean_ENMO'
    median.name = 'median_ENMO'
    var.name = 'var_ENMO'
    df = pd.merge(left=df, left_on='mergeid', right=mean, right_on='mergeid', how='left')
    df = pd.merge(left=df, left_on='mergeid', right=median, right_on='mergeid', how='left')
    df = pd.merge(left=df, left_on='mergeid', right=var, right_on='mergeid', how='left')
    return df


df.loc[:, 'age'] = df.replace({'-15. no information': -1})
df.loc[:, 'age'] = df['age'].astype(float)


df = calc_ENMO_stats(df)

df['physical_troubles'] = np.logical_or(~df['ph048dno'].astype(bool), ~df['ph049dno'].astype(bool)).astype(int)
df['activity'] = 2 * (4 - df['br015_']) + (4 - df['br016_'])
df['female'] = pd.to_numeric(df['female'].replace({'1. female': 1, '0. male': 0}))
df.loc[df['ph006d2'] == 1, 'hypertension'] = 'true'
df.loc[df['ph006d5'] == 1, 'diabetes'] = 'true'
df.loc[df['ph006d20'] == 1, 'rheuma'] = 'true'
df['healthy'] = np.logical_and(~df['ph006d2'].astype(bool), ~df['ph006d5'].astype(bool),
                               ~df['ph006d20'].astype(bool)).astype(int)
# df['hyp_dia'] = np.logical_and(df['ph006d2'], df['ph006d5'])
# df['hyp_rhe'] = np.logical_and(df['ph006d2'], df['ph006d20'])
# df['rhe_dia'] = np.logical_and(df['ph006d20'], df['ph006d5'])

df_anno = df.drop_duplicates(subset='mergeid', keep='first')

''' t-test for independent variables '''
''' using variance in ENMO of different diseases '''

h = df.query('hypertension == "true"')['var_ENMO']
d = df.query('diabetes == "true"')['var_ENMO']
r = df.query('rheuma == "true"')['var_ENMO']


t1 = ttest_ind(d, h, nan_policy='omit')
print(t1)
# Ttest_indResult(statistic=0.5167484502596387, pvalue=0.605333031667959)


t2 = ttest_ind(r, d, nan_policy='omit')
print(t2)
# Ttest_indResult(statistic=0.7794866173811178, pvalue=0.43569599798737746)


t3 = ttest_ind(r, h, nan_policy='omit')
print(t3)
# Ttest_indResult(statistic=1.6259285138342718, pvalue=0.10396784170886317)


''' Distribution of var_ENMO between hypertension and diabetes patients '''

fig = plt.Figure(figsize=(20, 10))

p_hypertension = plt.hist(df['var_ENMO'][df['hypertension'] == 'true'], label="hypertension",
                          density=True, alpha=0.75)
p_diabetes = plt.hist(df['var_ENMO'][df['diabetes'] == 'true'], label="diabetes", density=True, alpha=0.75)

plt.suptitle("Distribution of var_ENMO \n between hypertension and diabetes patients", fontsize=16)
plt.xlabel("variance of ENMO", fontsize=12)
plt.ylabel("Probability density", fontsize=12)

plt.text(1000, .0010,
         f"$\mu={df['var_ENMO'][df['hypertension'] == 'true'].mean(): .1f},"
         f"\sigma={df['var_ENMO'][df['hypertension'] == 'true'].std(): .1f}$", color='orange')

plt.text(1000, .0008,
         f"$\mu={df['var_ENMO'][df['diabetes'] == 'true'].mean(): .1f},"
         f"\sigma={df['var_ENMO'][df['diabetes'] == 'true'].std(): .1f}$", color='blue')

plt.savefig(os.path.join(plot_dir, "Distribution of var_ENMO between hypertension and diabetes patients.png"))
plt.show()

''' Distribution of var_ENMO between rheuma and diabetes patients '''

fig = plt.Figure(figsize=(20, 10))

p_rheuma = plt.hist(df['var_ENMO'][df['rheuma'] == 'true'], label="rheuma",
                          density=True, alpha=0.75)
p_diabetes = plt.hist(df['var_ENMO'][df['diabetes'] == 'true'], label="diabetes", density=True, alpha=0.75)

plt.suptitle("Distribution of var_ENMO \n between rheuma and diabetes patients", fontsize=16)
plt.xlabel("variance of ENMO", fontsize=12)
plt.ylabel("Probability density", fontsize=12)

plt.text(1000, .0010,
         f"$\mu={df['var_ENMO'][df['rheuma'] == 'true'].mean(): .1f},"
         f"\sigma={df['var_ENMO'][df['rheuma'] == 'true'].std(): .1f}$", color='orange')

plt.text(1000, .0008,
         f"$\mu={df['var_ENMO'][df['diabetes'] == 'true'].mean(): .1f},"
         f"\sigma={df['var_ENMO'][df['diabetes'] == 'true'].std(): .1f}$", color='blue')

plt.savefig(os.path.join(plot_dir, "Distribution of var_ENMO between rheuma and diabetes patients.png"))
plt.show()


''' Distribution of var_ENMO between rheuma and hypertension patients '''

fig = plt.Figure(figsize=(20, 10))

p_rheuma = plt.hist(df['var_ENMO'][df['rheuma'] == 'true'], label="rheuma",
                          density=True, alpha=0.75)
p_hypertension = plt.hist(df['var_ENMO'][df['hypertension'] == 'true'], label="diabetes", density=True, alpha=0.75)

plt.suptitle("Distribution of var_ENMO \n between rheuma and hypertension patients", fontsize=16)
plt.xlabel("variance of ENMO", fontsize=12)
plt.ylabel("Probability density", fontsize=12)

plt.text(1000, .0010,
         f"$\mu={df['var_ENMO'][df['rheuma'] == 'true'].mean(): .1f},"
         f"\sigma={df['var_ENMO'][df['rheuma'] == 'true'].std(): .1f}$", color='orange')

plt.text(1000, .0008,
         f"$\mu={df['var_ENMO'][df['hypertension'] == 'true'].mean(): .1f},"
         f"\sigma={df['var_ENMO'][df['hypertension'] == 'true'].std(): .1f}$", color='blue')

plt.savefig(os.path.join(plot_dir, "Distribution of var_ENMO between rheuma and hypertension patients.png"))
plt.show()