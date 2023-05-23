import mysql.connector as mariadb
import sys
import pandas as pd
import numpy as np
from functions import calculate_thi_score, calculate_minitq_score, calculate_tfi_score, calculate_whoqol_score, calculate_phq9_score, \
create_b10_categories, create_b22_categories, create_a15_categories, create_b11_categories, outliers_to_nan, check_outlier_q_a3, check_outlier_q_a4, \
check_q_a6_alcohol, corrX_new, cramers_v_corr, ordinal_corr, nominal_corr, nom_num_corr, nom_num_corr_drop_list, onehot_diff_Xtrain_Xtest, fix_treatment_codes, \
    plotGraph, get_age_ranges, audiological_loudness_and_masking,thi_diff_outliers
from scipy.stats import pointbiserialr 

from sklearn.preprocessing import LabelEncoder

try:
    conn = mariadb.connect(user="tdb_user",
        password= "Unit8",
        port=3493,
        database="tdb")
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

print(conn)
if(conn):
    print("Connection Succesful")
else:
    print("Connection failed")

    
"""""""""""""""""""""""""""""""""""""""""""""External ID"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# external id
external_id = pd.read_sql("""select p.id as patient_id, p.external_id as external
from patients p
left join sessions s on p.id = s.patient_id
left join session_content_description scd on s.session_id = scd.session_id
where scd.created_at>= '2021-02-10';""", con=conn)


# Create a new column in df2 with modified values of colA to match col1 in df1
external_id['external'] = external_id['external'].str.replace(' ', '')
external_id['result'] = external_id['external'].str.extract(r'(\d-\s*(\w+))|((\d{3}-\d{3}).+)')[1].fillna(external_id['external'].str.extract(r'(\d-\s*(\w+))|((\d{3}-\d{3}).+)')[3])
external_id['result'] = external_id['result'].fillna(external_id['external'])

"""""""""""""""""""""""""""""""""""""""""""""External ID"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




"""""""""""""""""""""""""""""""""""""""""""""Adherence to treatment"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

sc_adherence = pd.read_csv('./adherence_data/sc_adherence.csv', sep=';')
st_adherence = pd.read_csv('./adherence_data/st_adherence.csv', sep=';')
cbt_adherence = pd.read_csv('./adherence_data/CBT_adherence.csv', sep=';')
ha_adherence = pd.read_csv('./adherence_data/HA_adherence.csv', sep=';')


def st_sc_adherence_external(df, external_id):
    df['External Patient ID'] = df['external_id'].str.extract(r'(\d-\s*(\w+))|((\d{3}-\d{3}).+)')[1].fillna(df['external_id'].str.extract(r'(\d-\s*(\w+))|((\d{3}-\d{3}).+)')[3])
    df = pd.merge(df, external_id, left_on='External Patient ID', right_on='result')
    df = df.drop(columns=['External Patient ID','external','result', 'external_id'])

    return df
    
sc_adherence = st_sc_adherence_external(sc_adherence, external_id)   
st_adherence = st_sc_adherence_external(st_adherence, external_id)       
    


cbt_adherence['UNITI_ID'] = cbt_adherence['UNITI_ID'].str.replace('1-', '')
cbt_adherence.drop('center_name', inplace=True, axis=1)
cbt_adherence = pd.merge(cbt_adherence, external_id, left_on='UNITI_ID', right_on= 'result')
cbt_adherence.drop(columns=['UNITI_ID', 'external','result'], inplace=True)

mask = ha_adherence['external_id'].str.contains(r'^\d{3}-\d{3}-\d{3}\.\d$')
ha_adherence.loc[mask, 'UNITI_ID_short'] = ha_adherence.loc[mask, 'external_id'].str[:7]

ha_adherence.loc[[74,75,76], 'UNITI_ID_short'] = ha_adherence.loc[[74,75,76], 'external_id']
ha_adherence.drop('external_id', axis=1, inplace=True)
ha_adherence = pd.merge(ha_adherence, external_id, left_on='UNITI_ID_short', right_on='result')
ha_adherence.drop(columns=['UNITI_ID_short', 'external','result'], inplace=True)




"""""""""""""""""""""""""""""""""""""""""""""Adherence to treatment"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""Hearing Aid Indication"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   
#laura's csv

laura_csv = pd.read_csv('./CRF_140223.csv', sep=';', usecols=['External Patient ID', 
                                                                            'Hearing Aid Indication'])
laura_csv.columns



# Merge two dataframes on the new column colA_new
hearing_indication = pd.merge(laura_csv, external_id, left_on='External Patient ID', right_on='result')


hearing_indication = hearing_indication.drop(columns=['External Patient ID','external','result'])

# # Save Hearing Aid Indication
hearing_indication.to_pickle('flask/hearing_aid.pkl')



"""""""""""""""""""""""""""""""""""""""""""""Hearing Aid Indication"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""Leuven patients"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Patients to exclude if they are from Leuven
leuven_csv = pd.read_csv('./Leuven_SCED_external-ids.csv', sep=';')
leuven_csv['external'] = leuven_csv['external'].str.extract(r'(\d-\s*(\w+))|(\d-\s*(\w+)-)|((\d{3}-\d{3}).+)')[1].fillna(external_id['external'].str.extract(r'(\d-\s*(\w+))|((\d{3}-\d{3}).+)')[3])
leuven_csv = pd.merge(external_id, leuven_csv, left_on='result', right_on='external')

leuven_csv = leuven_csv.drop(columns=['external_x','external_y','result'])

"""""""""""""""""""""""""""""""""""""""""""""Exclude Leuven patients"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""AMLRs and ABRs"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# AMLRs and ABRs
abr_left_80 = pd.read_csv('./rano_amlr_abr/my_annotation_80L.csv', sep=';')
abr_left_90 = pd.read_csv('./rano_amlr_abr/my_annotation_90L.csv', sep=';')
abr_right_80 = pd.read_csv('./rano_amlr_abr/my_annotation_80R.csv', sep=';')
abr_right_90 = pd.read_csv('./rano_amlr_abr/my_annotation_90R.csv', sep=';')
amlr_right_70 = pd.read_csv('./rano_amlr_abr/my_annotation_70R.csv', sep=';')
amlr_left_70 = pd.read_csv('./rano_amlr_abr/my_annotation_70L.csv', sep=';')

def fix_external_id(df):
    df['patient_id'] = df['patient_id'].replace(r'^\d+-', '', regex=True) # remove any single digit and hyphen at the start
    df['patient_id'] = df['patient_id'].replace(r'^UNITI ', '', regex=True) # remove 'UNITI ' at the start    
    return df

abr_left_80 = fix_external_id(abr_left_80)
abr_left_90 = fix_external_id(abr_left_90)
abr_right_80 = fix_external_id(abr_right_80)
abr_right_90 = fix_external_id(abr_right_90)
amlr_right_70 = fix_external_id(amlr_right_70)
amlr_left_70 = fix_external_id(amlr_left_70)

abr_amlr_list = [amlr_right_70, amlr_left_70]

# Find the dataframe with the most rows
max_rows = max([df.shape[0] for df in abr_amlr_list])

# Merge dataframes based on "id" column
merged_df = pd.DataFrame({'patient_id': amlr_left_70['patient_id']})
for df in abr_amlr_list:
    if 'patient_id' in df.columns:
        df = df.drop_duplicates(subset=['patient_id'])
        if df.shape[0] < max_rows:
            temp_df = pd.DataFrame({'patient_id': [None] * (max_rows - df.shape[0])})
            temp_df = pd.concat([temp_df, df], axis=0)
            temp_df = temp_df.reset_index(drop=True)
            df = temp_df
        merged_df = pd.merge(merged_df, df, on='patient_id', how='left')


merged_df = pd.merge(merged_df, external_id, left_on='patient_id', right_on='result')
merged_df = merged_df.drop(columns=['patient_id_x','external','result'])
merged_df = merged_df.rename(columns={'patient_id_y': 'patient_id'})

def to_float(x):
    if isinstance(x, str) and ',' in x:
        return float(x.replace(',', '.'))
    else:
        return x

merged_df = merged_df.applymap(to_float)
merged_df = merged_df.astype(float)

"""""""""""""""""""""""""""""""""""""""""""""AMLRs and ABRs"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""Genetics"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Genetics

genetics_table = pd.read_csv('./genetics.csv', sep=';')
genetics_table = genetics_table.T
genetics_table = genetics_table.reset_index()

genetics_table.columns = genetics_table.iloc[0]
genetics_table = genetics_table.iloc[1:]

genetics = pd.merge(external_id, genetics_table, left_on='result', right_on='SYMBOL')

genetics = genetics.drop(columns=['SYMBOL','external','result'])

"""""""""""""""""""""""""""""""""""""""""""""Genetics"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





"""""""""""""""""""""""""""""""""""""""""""""SQL Tables"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Get visit day from baseline and screening

visit_day_baseline = pd.read_sql("""select p.id as patient_id, sc.visit_day
from  session_content as sc
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null;""", con=conn)

visit_day_screening = pd.read_sql("""select p.id as patient_id, sc.visit_day
from  session_content as sc
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null;""", con=conn)

# Patients's date of birth and sex

patient_birthdate = pd.read_sql("""select P.id as patient_id, P.date_of_birth
from patients as P
where P.id is not null;""", con=conn)

# Patients's sex
patient_sex = pd.read_sql("""select P.id as patient_id, P.sex as patient_sex
from patients as P
where P.id is not null;""", con=conn)

# Esitsq table
esit_questions = pd.read_sql(""" select p.id as patient_id, esit.*
from esitsq as esit

left join session_content AS sc on esit.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and esit.created_at>= '2021-01-01';""", con = conn)

# THI scores baseline - final 
thi_score_baseline = pd.read_sql("""select p.id as patient_id, vTHI.score as thi_score_baseline
from v_thi as vTHI

left join session_content AS sc on vTHI.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and vTHI.created_at>='2021-01-01';""", con=conn)

thi_score_screening = pd.read_sql("""select p.id as patient_id, vTHI.score as thi_score_screening
from v_thi as vTHI

left join session_content AS sc on vTHI.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null and vTHI.created_at>='2021-01-01';""", con=conn)

thi_score_final = pd.read_sql("""select p.id as patient_id, vTHI.score as thi_score_final
from v_thi as vTHI

left join session_content AS sc on vTHI.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_final_wardround' and p.id is not null and vTHI.created_at>= '2021-07-12';""", con=conn)

# Minitq score
mini_tq_score = pd.read_sql("""select p.id as patient_id, vmini.score as mini_score
from v_mini_tq as vmini
left join session_content AS sc on vmini.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and vmini.created_at>='2021-01-01';""", con=conn)

mini_tq_score_screening = pd.read_sql("""select p.id as patient_id, vmini.score as mini_score_screening
from v_mini_tq as vmini
left join session_content AS sc on vmini.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null and vmini.created_at>='2021-01-01';""", con=conn)

# TFI score baseline and final

tfi_scores_baseline = pd.read_sql("""select p.id as patient_id, vtfi.score as tfi_score, vtfi.score_sleep, vtfi.score_sense_of_control,
       vtfi.score_relaxation, vtfi.score_quality_of_live, vtfi.score_intrusive,
       vtfi.score_emotional, vtfi.score_cognitive, vtfi.score_auditory
from v_tfi as vtfi
left join session_content AS sc on vtfi.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null;""", con=conn)

tfi_scores_screening = pd.read_sql("""select p.id as patient_id, vtfi.score as tfi_score_screen, vtfi.score_sleep as score_sleep_screen, vtfi.score_sense_of_control as score_sense_of_control_screen,
       vtfi.score_relaxation as score_relaxation_screen, vtfi.score_quality_of_live as score_quality_of_live_screen, vtfi.score_intrusive as score_intrusive_screen,
       vtfi.score_emotional as score_emotional_screen, vtfi.score_cognitive as score_cognitive_screen, vtfi.score_auditory as score_auditory_screen
from v_tfi as vtfi
left join session_content AS sc on vtfi.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null;""", con=conn)


tfi_score_final = pd.read_sql("""select p.id as patient_id, vtfi.score as tfi_score_final
from v_tfi as vtfi
left join session_content AS sc on vtfi.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_final_wardround' and p.id is not null and vtfi.created_at>= '2021-07-12';""", con=conn)

# WHOQOL domain scores
who_qol_scores = pd.read_sql("""select p.id as patient_id, vwho.domain1 as whoqol_dom1, vwho.domain2 as whoqol_dom2,
       vwho.domain3 as whoqol_dom3, vwho.domain4 as whoqol_dom4
from v_whoqol_bref as vwho
left join session_content AS sc on vwho.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and vwho.created_at>= '2021-01-01';""", con=conn)

who_qol_scores_screening = pd.read_sql("""select p.id as patient_id, vwho.domain1 as whoqol_dom1_screen, vwho.domain2 as whoqol_dom2_screen,
       vwho.domain3 as whoqol_dom3_screen, vwho.domain4 as whoqol_dom4_screen
from v_whoqol_bref as vwho
left join session_content AS sc on vwho.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null and vwho.created_at>= '2021-01-01';""", con=conn)

# PHQ9 score
phq9_score = pd.read_sql("""select p.id as patient_id, vphq9.score as phq9_score
from v_phq9 as vphq9
left join session_content AS sc on vphq9.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null;""", con=conn)

phq9_score_screening = pd.read_sql("""select p.id as patient_id, vphq9.score as phq9_score_screen
from v_phq9 as vphq9
left join session_content AS sc on vphq9.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null;""", con=conn)

# Mini soises
mini_soises_table = pd.read_sql("""select p.id as patient_id, mini_so.* from mini_soises as mini_so
left join session_content AS sc on mini_so.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null;""", con=conn)

mini_soises_table_screening = pd.read_sql("""select p.id as patient_id, mini_so.* from mini_soises as mini_so
left join session_content AS sc on mini_so.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null;""", con=conn)

# Tinnitus severity
tinnitus_severity_table = pd.read_sql("""select p.id as patient_id, TS.* from tinnitus_severity as TS
left join session_content AS sc on TS.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and TS.created_at>= '2021-01-01';""", con=conn)

tinnitus_severity_table_screening = pd.read_sql("""select p.id as patient_id, TS.* from tinnitus_severity as TS
left join session_content AS sc on TS.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null and TS.created_at>= '2021-01-01';""", con=conn)

# Treatment table
treatment_table = pd.read_sql("""select p.id as patient_id, scd.code_intervention_protocol
from patients p
left join sessions s on p.id = s.patient_id
left join session_content_description scd on s.session_id = scd.session_id
where scd.created_at>= '2021-02-10';""", con=conn)

# BFI2 table
bfi2_table = pd.read_sql("""select p.id as patient_id, vbfi2.agreeableness, vbfi2.conscientiousness, vbfi2.extraversion, vbfi2.neg_emotion,
       vbfi2.openness
from v_bfi2  as vbfi2
left join session_content AS sc on vbfi2.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and vbfi2.created_at >= '2021-01-01';""", con=conn)

# Guef table
guef_table = pd.read_sql("""select p.id as patient_id, guef.score as guef_score
from v_guef  as guef
left join session_content AS sc on guef.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and guef.created_at >='2021-01-01';""", con=conn)

#FTQ table
ftq_table = pd.read_sql("""select p.id as patient_id, FTQ.*
from ftq  as FTQ
left join session_content AS sc on FTQ.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and FTQ.created_at >='2021-01-01';""", con=conn)

audiological_baseline = pd.read_sql("""select p.id as patient_id, vaudio.*
from v_audiological_examination as vaudio
left join session_content AS sc on vaudio.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_baseline' and p.id is not null and vaudio.created_at>='2021-01-01';""", con=conn)

audiological_screening = pd.read_sql("""select p.id as patient_id, vaudio.*
from v_audiological_examination as vaudio
left join session_content AS sc on vaudio.session_content_id = sc.session_content_id
left join sessions AS s on sc.session_id = s.session_id
left join patients AS p on p.id = s.patient_id
where sc.type_name like 'session_content_screening' and p.id is not null and vaudio.created_at>='2021-01-01';""", con=conn)

"""""""""""""""""""""""""""""""""""""""""""""SQL Tables"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





"""""""""""""""""""""""""""""""""""""""""""""FTQ Table Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Drop columns of no interest
ftq_table.drop(['id', 'last_validation_status', 'session_content_id','filled_by_survey_id', 'created_at', 'updated_at', 'deleted_at'], inplace=True, axis=1)

# Drop rows with NaN values
ftq_table.dropna(subset=ftq_table.columns.difference(['patient_id']), how='all', inplace=True)

# reverse values of question_4 and question_6
ftq_table.question_4 = ftq_table.question_4.map({0:1, 1:0})
ftq_table.question_6 = ftq_table.question_6.map({0:1, 1:0})

# Sum all questions to get the total ftq score

ftq_table['ftq_score'] = ftq_table.iloc[:, 1:].sum(axis=1)

# Drop questions from ftq table
ftq_table.drop(ftq_table.columns[1:18], inplace=True, axis=1)

"""""""""""""""""""""""""""""""""""""""""""""FTQ Table Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





"""""""""""""""""""""""""""""""""""""""""""""Audiological combine baseline and screening"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Audiological 

audiological_cols = ['audiological_examination_id', 'session_content_id',
       'examination_date', 'date_unspecified',
       'left_frequency_loss_10', 'left_frequency_loss_11',
       'left_frequency_loss_12', 'left_frequency_loss_13',
       'left_frequency_loss_14', 
       'right_frequency_loss_10', 'right_frequency_loss_11',
       'right_frequency_loss_12', 'right_frequency_loss_13',
       'right_frequency_loss_14', 'survey_participation_appropriateness',
       'created_at', 'updated_at', 'deleted_at', 'last_validation_status',
       'filled_by_survey_id','legacy_residual_inhibition', 'legacy_duration', 'legacy_active',
       'residual_inhibitions']

# Drop columns of no interest from audiological table (baseline and screening)
audiological_baseline = audiological_baseline.drop(columns=audiological_cols, axis=1)
audiological_screening = audiological_screening.drop(columns=audiological_cols, axis=1)

"""Rename column names of audiological screening"""

rename_dict_audio_screen = {col: col + '_screen' for col in audiological_screening.columns if col != 'patient_id'}
audiological_screening = audiological_screening.rename(columns=rename_dict_audio_screen)


audiological_baseline_list = audiological_baseline.drop('patient_id', axis=1).columns.tolist()
audiologica_screening_list = audiological_screening.drop('patient_id', axis=1).columns.tolist()
merged_audiological = audiological_baseline.merge(audiological_screening, on='patient_id', how='outer')
merged_audiological[audiological_baseline_list] = np.where(pd.isna(merged_audiological[audiological_baseline_list]), merged_audiological[audiologica_screening_list], merged_audiological[audiological_baseline_list])
audiological_baseline = merged_audiological.drop(audiologica_screening_list, axis=1)

"""""""""""""""""""""""""""""""""""""""""""""Audiological combine baseline and screening"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





"""""""""""""""""""""""""""""""""""""""""""""Drop duplicates from all tables"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#drop duplicate rows for all tables

sql_tables = [esit_questions,audiological_baseline, audiological_screening, thi_score_baseline, thi_score_final, thi_score_screening, mini_tq_score, tfi_scores_baseline, tfi_score_final, 
              who_qol_scores, phq9_score, mini_soises_table, tinnitus_severity_table, treatment_table, bfi2_table, guef_table, ftq_table]

for table in sql_tables:
    table.drop_duplicates(subset='patient_id', keep='first', inplace=True)
    
"""""""""""""""""""""""""""""""""""""""""""""Drop duplicates from all tables"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""ESITSQ Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Drop columns of no interest

esit_questions.drop(['esitsq_id', 'session_content_id', 'created_at', 'updated_at', 'deleted_at', 'q_o1_handedness', 'q_o2_residence_country', 'q_o3_birth_country', 
                    'q_o4_marital_status', 'q_o5_economical_status', 'q_o6_employment', 'q_o7_night_work', 'q_o8_cigarettes', 'q_o9_coffee', 'q_o10_exercise', 'q_o11_meat', 
                    'q_o12_fish', 'q_o13_fruits', 'q_o14_vegetables', 'q_o15_mobile_calls', 'q_o16_headphones_music', 'q_o17_sleep', 'q_a10_none', 'q_a11_none', 'q_a14_none', 
                    'q_a15_none', 'q_a16_none','q_b9_none','q_b10_painkillers_text','q_b10_steroids_text','q_b10_antibiotics_text','q_b10_antidepressants_text',
                    'q_b10_none','q_b11_none','q_b18_none','q_b19_none', 'q_b21_none', 'q_b22_none','last_validation_status','q_b10_quinine_text','q_b10_diuretics_text',
                    'q_b13_other_text','q_b15_other_text','q_b16_other_text','filled_by_survey_id', 'q_a10_other1_temp_rel','q_a10_other2_temp_rel','q_a10_other3_temp_rel',
                    'q_a11_other1_temp_rel','q_a11_other2_temp_rel','q_a11_other3_temp_rel','q_a15_other1_temp_rel','q_a15_other2_temp_rel','q_a15_other3_temp_rel',
                    'q_a16_other1_temp_rel','q_a16_other2_temp_rel','q_a16_other3_temp_rel','q_a10_other1', 'q_a10_other2', 'q_a10_other3', 'q_a11_other1', 'q_a11_other2', 
                    'q_a11_other3', 'q_a16_other1', 'q_a16_other2','q_a16_other3', 'q_b9_other1','q_b9_other2', 'q_b9_other3','q_b18_other1','q_b18_other2','q_b18_other3',
                    'q_b19_other1','q_b19_other2','q_b19_other3', 'q_b21_other1', 'q_b21_other2', 'q_b21_other3','q_a15_other1', 'q_a15_other2', 'q_a15_other3', 'q_b10_other1',
                    'q_b10_other2','q_b10_other3', 'q_b11_other1', 'q_b11_other2', 'q_b11_other3', 'q_b22_other1', 'q_b22_other2', 'q_b22_other3'], inplace=True, axis=1)

esit_columns = pd.Series(esit_questions.columns)

# create a list of columns to replace nan with 0 and -1 to 0 for _temp_rel
import re

nan_pattern = re.compile(r'^q_a10_|^q_a11_|^q_a14_|^q_a15_|^q_a16_|^q_b9_|^q_b10_|^q_b18_|^q_b19_|^q_b21_')
temp_rel_pattern = re.compile(r'_temp_rel$')

# Replace NaN values with 0 in columns matching nan_pattern
cols_replace_nan = [col for col in esit_questions.columns if nan_pattern.match(col)]
esit_questions[cols_replace_nan] = esit_questions[cols_replace_nan].fillna(0)

# Replace -1 with 0 in columns matching temp_rel_pattern
cols_replace_minus_one = [col for col in esit_questions.columns if temp_rel_pattern.search(col)]
esit_questions[cols_replace_minus_one] = esit_questions[cols_replace_minus_one].replace(-1, 0)

# Drop zero columns
esit_questions = esit_questions.loc[:, (esit_questions != 0).any(axis=0)]

"""""""""""""""""""""""""""""""""""""""""""""ESITSQ Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""Rename certain columns - Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# mini_soises Table  - Baseline

mini_soises_table = mini_soises_table[["patient_id", "question_1", "question_2", "question_3", "question_4", "question_5", "question_6", "question_7", 
                                       "question_8", "question_9", "question_10", "question_11"]]

"""Rename column names to mini_soises"""

mini_dict = {f'question_{number}': f'mini_soises_{number}' for number in range(1, len(mini_soises_table.columns))}
        
mini_soises_table.rename(columns= mini_dict, inplace=True)


# mini_soises Table  - Screening

mini_soises_table_screening = mini_soises_table_screening[["patient_id", "question_1", "question_2", "question_3", "question_4", "question_5", "question_6", "question_7", 
                                       "question_8", "question_9", "question_10", "question_11"]]

"""Rename column names to mini_soises"""

mini_dict_screen = {f'question_{number}': f'mini_soises_screen_{number}' for number in range(1, len(mini_soises_table_screening.columns))}
        
mini_soises_table_screening.rename(columns= mini_dict_screen, inplace=True)

    
#Tinnitus severity Table - Baseline

tinnitus_severity_table = tinnitus_severity_table[["patient_id", "question_1", "question_2", "question_3", "question_4", "question_5", "question_6"]]

""" Rename column names to tinn_sever_*"""

tinn_sever_dict = {f'question_{number}': f'tinn_sever_{number}' for number in range(1, len(tinnitus_severity_table.columns))}

tinnitus_severity_table.rename(columns = tinn_sever_dict, inplace=True)


#Tinnitus severity Table - Screening

tinnitus_severity_table_screening = tinnitus_severity_table_screening[["patient_id", "question_1", "question_2", "question_3", "question_4", "question_5", "question_6"]]

""" Rename column names to tinn_sever_*"""

tinn_sever_dict_screen = {f'question_{number}': f'tinn_sever_screen_{number}' for number in range(1, len(tinnitus_severity_table_screening.columns))}

tinnitus_severity_table_screening.rename(columns = tinn_sever_dict_screen, inplace=True)

# Audiological table - Screening 

rename_dict = {col: col + '_screen' for col in audiological_screening.columns if col != 'patient_id'}

audiological_screening = audiological_screening.rename(columns=rename_dict)

# Visit day -  Screening

rename_dict_visit = {col: col + '_screen' for col in visit_day_screening.columns if col != 'patient_id'}
visit_day_screening = visit_day_screening.rename(columns=rename_dict_visit)

"""""""""""""""""""""""""""""""""""""""""""""Rename certain columns - Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""Remove Leuven patients"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Delete patients that are from Leuven
merged_baseline = pd.merge(leuven_csv, thi_score_final, on='patient_id', how='right', indicator=True)
thi_score_final = merged_baseline[merged_baseline['_merge'] == 'right_only'].drop('_merge', axis=1)

"""""""""""""""""""""""""""""""""""""""""""""Remove Leuven patients"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""Replace missing values from baseline with values from screening"""""""""""""""""""""""""""""""""""""""""""""

# Visit day
# drop duplicates from visit day
visit_duplicates = [visit_day_baseline, visit_day_screening]
for visit_dup in visit_duplicates:
    visit_dup.drop_duplicates(subset='patient_id', keep='last', inplace=True)
merged_visit_day = visit_day_baseline.merge(visit_day_screening, on='patient_id', how = 'outer')
merged_visit_day['visit_day'] = np.where(pd.isna(merged_visit_day['visit_day']), merged_visit_day['visit_day_screen'], merged_visit_day['visit_day'])
visit_day_baseline = merged_visit_day.drop('visit_day_screen', axis=1)

visit_day_baseline['visit_day'] = pd.to_datetime(visit_day_baseline['visit_day'])
visit_day_baseline['visit_year'] = visit_day_baseline['visit_day'].dt.year
visit_day_baseline.drop('visit_day', axis=1, inplace=True)

#THI
merged_thi = thi_score_baseline.merge(thi_score_screening, on='patient_id', how='outer')
merged_thi['thi_score_baseline'] = np.where(pd.isna(merged_thi['thi_score_baseline']), merged_thi['thi_score_screening'], merged_thi['thi_score_baseline'])
thi_score_baseline = merged_thi.drop('thi_score_screening', axis=1)

# Minitq
merged_minitq = mini_tq_score.merge(mini_tq_score_screening, on='patient_id', how='outer')
merged_minitq['mini_score'] = np.where(pd.isna(merged_minitq['mini_score']), merged_minitq['mini_score_screening'], merged_minitq['mini_score'])
mini_tq_score = merged_minitq.drop('mini_score_screening', axis=1)

# TFI
tfi_baseline_list = ['tfi_score', 'score_sleep', 'score_sense_of_control',
       'score_relaxation', 'score_quality_of_live', 'score_intrusive',
       'score_emotional', 'score_cognitive', 'score_auditory']

tfi_screen_list = ['tfi_score_screen', 'score_sleep_screen',
       'score_sense_of_control_screen', 'score_relaxation_screen',
       'score_quality_of_live_screen', 'score_intrusive_screen',
       'score_emotional_screen', 'score_cognitive_screen',
       'score_auditory_screen']
merged_tfi = tfi_scores_baseline.merge(tfi_scores_screening, on='patient_id', how='outer')
merged_tfi[tfi_baseline_list] = np.where(pd.isna(merged_tfi[tfi_baseline_list]), merged_tfi[tfi_screen_list], merged_tfi[tfi_baseline_list])
tfi_scores_baseline = merged_tfi.drop(tfi_screen_list, axis=1)

# Whoqol

whoqol_baseline_list = ['whoqol_dom1', 'whoqol_dom2', 'whoqol_dom3',
       'whoqol_dom4']

whoqol_screening_list = ['whoqol_dom1_screen', 'whoqol_dom2_screen',
       'whoqol_dom3_screen', 'whoqol_dom4_screen']

merged_whoqol = who_qol_scores.merge(who_qol_scores_screening, on='patient_id', how='outer')
merged_whoqol[whoqol_baseline_list] = np.where(pd.isna(merged_whoqol[whoqol_baseline_list]), merged_whoqol[whoqol_screening_list], merged_whoqol[whoqol_baseline_list])
who_qol_scores = merged_whoqol.drop(whoqol_screening_list, axis=1)

# PHQ9

merged_phq9 = phq9_score.merge(phq9_score_screening, on='patient_id', how='outer')
merged_phq9['phq9_score'] = np.where(pd.isna(merged_phq9['phq9_score']), merged_phq9['phq9_score_screen'], merged_phq9['phq9_score'])
phq9_score = merged_phq9.drop('phq9_score_screen', axis=1)

# Mini soises
mini_soises_baseline_list = ['mini_soises_1', 'mini_soises_2', 'mini_soises_3',
       'mini_soises_4', 'mini_soises_5', 'mini_soises_6', 'mini_soises_7',
       'mini_soises_8', 'mini_soises_9', 'mini_soises_10', 'mini_soises_11']

mini_soises_screening_list = ['mini_soises_screen_1', 'mini_soises_screen_2', 'mini_soises_screen_3',
'mini_soises_screen_4', 'mini_soises_screen_5', 'mini_soises_screen_6',
'mini_soises_screen_7', 'mini_soises_screen_8', 'mini_soises_screen_9',
'mini_soises_screen_10', 'mini_soises_screen_11']

merged_mini_soises = mini_soises_table.merge(mini_soises_table_screening, on='patient_id', how='outer')
merged_mini_soises[mini_soises_baseline_list] = np.where(pd.isna(merged_mini_soises[mini_soises_baseline_list]), merged_mini_soises[mini_soises_screening_list], merged_mini_soises[mini_soises_baseline_list])
mini_soises_table = merged_mini_soises.drop(mini_soises_screening_list, axis=1)

# Tinnitus Severity

tinn_sever_baseline_list = ['tinn_sever_1', 'tinn_sever_2', 'tinn_sever_3',
       'tinn_sever_4', 'tinn_sever_5', 'tinn_sever_6']

tinn_sever_screening_list = ['tinn_sever_screen_1',
       'tinn_sever_screen_2', 'tinn_sever_screen_3', 'tinn_sever_screen_4',
       'tinn_sever_screen_5', 'tinn_sever_screen_6']

merged_tinn_sever = tinnitus_severity_table.merge(tinnitus_severity_table_screening, on='patient_id', how='outer')
merged_tinn_sever[tinn_sever_baseline_list] = np.where(pd.isna(merged_tinn_sever[tinn_sever_baseline_list]), merged_tinn_sever[tinn_sever_screening_list], merged_tinn_sever[tinn_sever_baseline_list])
tinnitus_severity_table = merged_tinn_sever.drop(tinn_sever_screening_list, axis=1)


# Replace missing values for sex from esit_sq questionnaire using information about sex from patients's table
esit_sex = pd.DataFrame(esit_questions[['patient_id','q_a2_sex']])
merged_sex = esit_sex.merge(patient_sex, on='patient_id', how='outer')
merged_sex['q_a2_sex'] = np.where(pd.isna(merged_sex['q_a2_sex']), merged_sex['patient_sex'], merged_sex['q_a2_sex'])
esit_sex = merged_sex.drop('patient_sex', axis=1)

esit_questions.drop('q_a2_sex', axis=1, inplace=True)
esit_questions = pd.merge(esit_questions, esit_sex, on='patient_id', how='left')


# Replace missing values for age from esit_sq_questionnaire using information about age from patients's table
patient_birthdate['date_of_birth'] = pd.to_datetime(patient_birthdate['date_of_birth'])
patient_birthdate['year_of_birth'] = patient_birthdate['date_of_birth'].dt.year
patient_birthdate.drop('date_of_birth', axis=1, inplace=True)

merged_age = pd.merge(visit_day_baseline, patient_birthdate, on='patient_id', how='left')
merged_age['Age'] = merged_age['visit_year'] - merged_age['year_of_birth']
merged_age.drop(['visit_year', 'year_of_birth'], axis=1, inplace=True)


esit_questions = pd.merge(esit_questions, merged_age, on='patient_id', how='left')
esit_questions['q_a1_age'] = np.where(pd.isna(esit_questions['q_a1_age']), esit_questions['Age'], esit_questions['q_a1_age'])
esit_questions.drop('Age', axis=1, inplace=True)

"""""""""""""""""""""""""""""""""""""""""""""Replace missing values from baseline with values from screening"""""""""""""""""""""""""""""""""""""""""""""





"""""""""""""""""""""""""""""""""""""""""""""Transform whoqol values from 4-20 to 0-100 scale"""""""""""""""""""""""""""""""""""""""""""""

# Transform whoqol values from 4-20 to 0-100 scale
max_score = who_qol_scores.max().max()  # compute max score dynamically

# compute transformed scores for each domain
for col in who_qol_scores.columns:
    if col !='patient_id':
        who_qol_scores[col] = who_qol_scores[col].apply(lambda x: (x - 4) / (max_score - 4) * 100)
        
"""""""""""""""""""""""""""""""""""""""""""""Transform whoqol values from 4-20 to 0-100 scale"""""""""""""""""""""""""""""""""""""""""""""        


        
        
        
"""""""""""""""""""""""""""""""""""""""""""Merge all tables under the baseline_feature dataframe"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Merge all dataframes based on patient_id from thi_score_final

features_created_list = [treatment_table, esit_questions, tfi_scores_baseline, 
                         thi_score_baseline, mini_tq_score, who_qol_scores, mini_soises_table, tinnitus_severity_table, 
                         phq9_score, hearing_indication, ftq_table, bfi2_table, guef_table, thi_score_final]

# Find the dataframe with the most rows
max_rows_features = max([df.shape[0] for df in features_created_list])

# Merge dataframes based on "id" column
baseline_features = pd.DataFrame({'patient_id': thi_score_final['patient_id']})
for df in features_created_list:
    if 'patient_id' in df.columns:
        df = df.drop_duplicates(subset=['patient_id'])
        if df.shape[0] < max_rows_features:
            temp_df = pd.DataFrame({'patient_id': [None] * (max_rows_features - df.shape[0])})
            temp_df = pd.concat([temp_df, df], axis=0)
            temp_df = temp_df.reset_index(drop=True)
            df = temp_df
        baseline_features = pd.merge(baseline_features, df, on='patient_id', how='left')
        
"""""""""""""""""""""""""""""""""""""""""""Merge all tables under the baseline_feature dataframe"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        
        




"""""""""""""""""""""""""""""""""""""""""""Transform -1 and outliers to Nan values"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""      

baseline_features.replace(-1, np.nan, inplace=True)

"""""""""""""""""""""""""""""""""""""""""""Transform -1 and outliers to Nan values"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""     





"""""""""""""""""""""""""""""""""""""""""""""""""""""""Outlier detection"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  

#List of features to check for outliers

zero_to_one_features = ['q_a8_fa', 'q_a8_mo', 'q_a10_', 'q_a11_', 'q_a14_', 'q_a15_', 'q_a16_', 'q_b9_', 'q_b10_', 'q_b18_', 'q_b19_', 'q_b21_']
one_to_two_features = ['q_b2_', 'q_b6_', 'q_b7_', 'q_b17_']
one_to_three_features = ['q_a9', 'q_b12', 'q_b14']
one_to_four_features = ['q_b4_', 'q_b20_']
one_to_five_features = ['q_a12_', 'q_a13_', 'q_a17_', 'q_b1_', 'q_b13_', 'q_b16_']
one_to_eight_features = ['q_b15_']

""" Dictionary  --> e.g key = '[0, 1]', value = [zero_to_one_features, [0, 1]] """

check_outliers_dict = {
    '[0, 1]': [zero_to_one_features, [0, 1]],
    '[1, 2]': [one_to_two_features, [1, 2]],
    '[1, 3]': [one_to_three_features, [1, 3]],
    '[1, 4]': [one_to_four_features, [1, 4]],
    '[1, 5]': [one_to_five_features, [1, 5]],
    '[1, 8]': [one_to_eight_features, [1, 8]]}


"""In the function below we pass as arguments the list of questions
that will be checked for outliers and the range e.g [0,1] which is the range 
that is expected for each question """

for value in check_outliers_dict.values():
        baseline_features = outliers_to_nan(baseline_features, value[0], value[1])

"""Replace outlier values from q_a3_height and q_a4_weight features with nan"""

baseline_features.q_a3_height = check_outlier_q_a3(baseline_features.q_a3_height)

baseline_features.q_a4_weight = check_outlier_q_a4(baseline_features.q_a4_weight)


""" q_a6_alcohol --> Replace ml to average no of drinks and remove potential outliers"""

baseline_features.q_a6_alcohol = check_q_a6_alcohol(baseline_features.q_a6_alcohol)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""Outlier detection"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  





"""""""""""""""""""""""""""""""""""""""""""Code Intervention Protocol - Preprocessing""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

# Drop rows that code intervention protocol is NaN
baseline_features = baseline_features[baseline_features['code_intervention_protocol'].notna()]

# Drop treatments of no interest
treatments_to_drop =['unitips','demo-prescreening','1-075', '34','thi','UNITIPS', '34-F5X5YM','unitips_example', 
                     '7','bas2-gra','milou-studie', '001-080', '001-081', 'uniti-big-2021', 'uniti-big-2022', 'bidt-demo','001-082']

for treatment in treatments_to_drop:
    
    baseline_features = baseline_features[baseline_features.code_intervention_protocol != treatment]



#reset index
baseline_features.reset_index(inplace=True)
baseline_features.drop(['index'], inplace=True, axis=1)

# Treatment column. Replace wrongly imported labels and assign numerical values to labels
baseline_features = fix_treatment_codes(baseline_features)

"""""""""""""""""""""""""""""""""""""""""""Code Intervention Protocol - Preprocessing"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""







"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Audiological Dataframe"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Merge audiological dataset with the baseline_feature on a new dataframe and preprocess audiological dataset - I did this here so that I keep only the patients that are relevant in the audiological dataframe
# Set the patient_id column as the index in both dataframes


audiological_baseline = audiological_baseline.set_index("patient_id")
baseline_features = baseline_features.set_index("patient_id")

# Join audiological and baseline features dataframes
merged_audiol_feat_baseline = audiological_baseline.join(baseline_features, how="inner")

# Reset the index to move patient_id back to a regular column
audiological_baseline = audiological_baseline.reset_index()
baseline_features = baseline_features.reset_index()
merged_audiol_feat_baseline  = merged_audiol_feat_baseline.reset_index()

audiological_baseline = merged_audiol_feat_baseline.loc[:,audiological_baseline.columns]

# Drop left and right frequency_loss_1 columns due to a large number of nan values
audiological_baseline.drop(['left_frequency_loss_1', 'right_frequency_loss_1'], inplace=True, axis=1)

# Replace 999 and 888 to nan values in columns start with left_frequency_loss_
# get the column names that start with 'right_frequency_loss_' or 'left_frequency_loss_'
freq_loss_cols = [col for col in audiological_baseline.columns if col.startswith(('right_frequency_loss_', 'left_frequency_loss_'))]

# replace 999 and 888 values with NaN in the selected columns
audiological_baseline[freq_loss_cols] = audiological_baseline[freq_loss_cols].applymap(lambda x: np.nan if x in [999, 888] else x)

# Replace potential -1 values with nan in the tinnitus_matching column, 'left_hearing_loss', 'right_frequency_loss', 'combined_residual_inhibition', 'left_residual_inhibition', 'right_residual_inhibition' columns
minus_one_to_nan_list = ['tinnitus_matching', 'left_hearing_loss', 'right_hearing_loss', 'combined_residual_inhibition', 'left_residual_inhibition','right_residual_inhibition']
audiological_baseline[minus_one_to_nan_list] = audiological_baseline[minus_one_to_nan_list].replace(-1, np.nan)

# select rows where tinnitus_matching == 2 and replace minimum values of the 4 frequency columns with the maximum (this is implemented in cases where the left and right min have different values from the left and right max)

for i, row in audiological_baseline[audiological_baseline['tinnitus_matching'] == 2].iterrows():
    # get the values for the left and right matching frequencies
    left_min = row['left_matching_frequency_min']
    right_min = row['right_matching_frequency_min']
    
    # compare the values and replace if necessary
    if left_min < right_min:
        audiological_baseline.at[i, 'left_matching_frequency_min'] = right_min
        audiological_baseline.at[i, 'left_matching_frequency_max'] = right_min
    if right_min < left_min:
        audiological_baseline.at[i, 'right_matching_frequency_min'] = left_min
        audiological_baseline.at[i, 'right_matching_frequency_max'] = left_min


# Create binary columns (based on the unique requency values) for right and left matching frequency min and max columns

unique_values = sorted(set(audiological_baseline[['right_matching_frequency_min', 'right_matching_frequency_max', 'left_matching_frequency_min', 'left_matching_frequency_max']].values.flatten()))
unique_values = [x for x in unique_values if not np.isnan(x)]

# create new columns for each unique value
for val in unique_values:
    audiological_baseline[str(val)] = np.where(((audiological_baseline['tinnitus_matching'] == 1) & ((audiological_baseline['left_matching_frequency_max'] == val) | 
                                                                                                     (audiological_baseline['left_matching_frequency_min'] == val))), 1, 0)
    audiological_baseline[str(val)] = np.where(((audiological_baseline['tinnitus_matching'] == 0) & ((audiological_baseline['right_matching_frequency_max'] == val) | 
                                                                                                     (audiological_baseline['right_matching_frequency_min'] == val))), 1, audiological_baseline[str(val)])
    audiological_baseline[str(val)] = np.where(((audiological_baseline['tinnitus_matching'] == 2) & ((audiological_baseline['right_matching_frequency_max'] == val) | 
                                                                                                    (audiological_baseline['right_matching_frequency_min'] == val) | 
                                                                                                    (audiological_baseline['left_matching_frequency_max'] == val) | 
                                                                                                    (audiological_baseline['left_matching_frequency_min'] == val))), 1, audiological_baseline[str(val)])
    audiological_baseline[str(val)] = np.where(((np.isnan(audiological_baseline['right_matching_frequency_max']) | np.isnan(audiological_baseline['right_matching_frequency_min']) 
                                                 | np.isnan(audiological_baseline['left_matching_frequency_max']) | np.isnan(audiological_baseline['left_matching_frequency_min'])) & 
                                                (audiological_baseline[['right_matching_frequency_max', 'right_matching_frequency_min', 'left_matching_frequency_max', 'left_matching_frequency_min']] == val).any(axis=1)), 1, 
                                               audiological_baseline[str(val)])

# Drop frequency columns
audiological_baseline.drop(columns=['right_matching_frequency_min', 'right_matching_frequency_max', 'left_matching_frequency_min', 'left_matching_frequency_max'], inplace=True)

            

# Preprocess left and right matching loudness columns            
audiological_baseline = audiological_loudness_and_masking(audiological_baseline, 'left_matching_loudness', 'right_matching_loudness')


# Preprocess left and right minimal masking level columns            
audiological_baseline = audiological_loudness_and_masking(audiological_baseline, 'left_minimal_masking_level', 'right_minimal_masking_level') 



# Replace NaN values in the right and left matching type with a new category (e.g. 4) when tinnitus_matching is not NaN
audiological_baseline.loc[audiological_baseline['tinnitus_matching'].notna(), ['right_matching_type', 'left_matching_type']] = \
    audiological_baseline.loc[audiological_baseline['tinnitus_matching'].notna(), ['right_matching_type', 'left_matching_type']].fillna(4)


# Residual Preprocessing - Replace NaN values in the combined, right and left residual inhibition with a new category [3] when they have NaN values
# If all three columns are nan then (if residual_inhibition=1 => combined_residual_inhibition=3), (if residual_inhibition=0 => left_residual_inhibition=3 and right_residual_inhibition=3)
# If combined_residual_inhibition !=nan => left_residual_inhibition=3 and right_residual_inhibition=3
# if left_residual_inhibition !=nan => combined_residual_inhibition =3 and right_residual_inhibition=3
# if right_residual_inhibition !=nan => combined_residual_inhibition =3 and left_residual_inhibition=3 

mask1 = audiological_baseline['residual_inhibition'] == 1
mask2 = audiological_baseline['residual_inhibition'] == 0
mask3 = audiological_baseline[['combined_residual_inhibition', 'left_residual_inhibition', 'right_residual_inhibition']].notna().any(axis=1)


audiological_baseline.loc[mask1 & ~mask3, 'combined_residual_inhibition'] = 3
audiological_baseline.loc[mask2 & ~mask3, ['left_residual_inhibition', 'right_residual_inhibition']] = 3

audiological_baseline.loc[mask1 & mask3, 'combined_residual_inhibition'] = 3
audiological_baseline.loc[mask3, ['left_residual_inhibition', 'right_residual_inhibition']] = audiological_baseline.loc[mask3, ['left_residual_inhibition', 'right_residual_inhibition']].fillna(3)

# Drop residual columns until I preprocess those columns
# audiological_baseline.drop(['combined_residual_inhibition','left_residual_inhibition','right_residual_inhibition'], inplace=True, axis=1)

      
# Merge audiological_baseline to baseline_features
baseline_features = pd.merge(audiological_baseline, baseline_features, on='patient_id', how='left')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Audiological Dataframe"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Assign numerical values to treatment_code, language and q_a2_sex feature"""""""""""""""""""""""""""""""""""""""""""

"""For sex """
sex  = ['male', 'female', 'intersex']

for element in range(len(sex)):
    baseline_features.replace(sex[element], element, inplace=True)
    
"""For language """
language = ['de', 'el', 'es', 'en', 'nl', 'fr']    
for lang in range(len(language)):
    baseline_features.replace(language[lang], lang, inplace=True)
 
"""For code interventions """
code_interventions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

for intervention in range(len(code_interventions)):
    baseline_features.replace(code_interventions[intervention], intervention, inplace=True)
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Assign numerical values to treatment_code, language and q_a2_sex feature"""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Drop large amount of NaN or zero values from columns and rows"""""""""""""""""""""""""""""""""""""""""""
# Drop columns that all values are nan or zero  
baseline_features = baseline_features.loc[:, (~baseline_features.isin([np.nan])).any()]  


#Keep columns with NaN < 0.6 / per column
baseline_features = baseline_features.loc[:, baseline_features.isnull().mean() < 0.8]

#Keep rows with NaN < 0.9 / per row
baseline_features = baseline_features.dropna(thresh =len( baseline_features.columns)*0.9)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Drop large amount of NaN or zero values from columns and rows"""""""""""""""""""""""""""""""""""""""""""

import matplotlib.pyplot as plt
baseline_columns = baseline_features.columns
thi_diff = baseline_features.loc[:, ['patient_id', 'thi_score_baseline', 'thi_score_final']]

merged_cbt = pd.merge(st_adherence, thi_diff, on='patient_id')
merged_cbt['thi_diff'] = merged_cbt['thi_score_baseline'] - merged_cbt['thi_score_final']

import seaborn as sns

sns.lmplot(x='n_st_sounds', y='thi_diff', data=merged_cbt, x_estimator=np.mean)


plt.scatter(merged_cbt['n_st_sounds'], merged_cbt['thi_diff'])
#plt.plot(merged_cbt['cbt_sessions'], merged_cbt['thi_diff'], '-o')
plt.xlabel('Number of CBT sessions')
plt.ylabel('THI score difference (final - baseline)')
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Split to Train and Test set"""""""""""""""""""""""""""""""""""""""""""

# Drop patient_id column
baseline_features = baseline_features.drop('patient_id', 1)

# Splitting baseline_features to X and y
X = baseline_features.iloc[:, :-1]
y = baseline_features.iloc[:, -1]    


# Splitting data to training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=baseline_features['code_intervention_protocol'], random_state=42)
    
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Split to Train and Test set"""""""""""""""""""""""""""""""""""""""""""







"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Check relationship between treatment and thi score change and remove potential outliers for X_train,X_test"""""""""""""""""""""""""""""""""""""""""""

X_train.name = 'X_train'
X_test.name = 'X_test'

X_train, y_train = thi_diff_outliers(X_train, y_train, 'code_intervention_protocol')
X_test, y_test = thi_diff_outliers(X_test, y_test, 'code_intervention_protocol')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Check relationship between treatment and thi score change and remove potential outliers for X_train,X_test"""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Convert categories to categorical types"""""""""""""""""""""""""""""""""""""""""""

# # convert to categorical dtype variables that are categorical
# X_train_columns = pd.Series(X_train.columns)

# categories = ['participant_lang','q_a2_sex', 'q_a7_smoking_status', 'q_a8_father', 'q_a8_mother', 'q_a9_vertigo', 'q_a10_', 'q_a11_', 
#               'q_a12_ext_sounds_prob', 'q_a13_hearing_difficulty', 'q_a14_', 'q_a15_', 'q_a16_', 'q_a17_tinnitus', 
#               'q_b1_frequency', 'q_b2_day_pattern', 'q_b4_bother', 'q_b6_number_sounds', 'q_b7_onset_char', 'q_b9_',
#               'q_b10_', 'q_b12_loudness_changes', 'q_b13_quality', 'q_b14_pitch', 'q_b15_localisation', 'q_b16_rhythmic',
#               'q_b17_objective', 'q_b18_', 'q_b19_', 'q_b20_tinnitus_healthcare', 'q_b21_', 'mini_soises_', 'tinn_sever_',
#                'b11_', 'b22_', 'b10_', 'a15_', 'Hearing Aid Indication', 'code_intervention_protocol', 'Age category', 'left_hearing_loss',
#                'right_hearing_loss', 'tinnitus_matching', 'right_matching_type','left_matching_type', 'residual_inhibition', 
#                'combined_residual_inhibition', 'left_residual_inhibition','right_residual_inhibition', '18000.0',
#                '125.0', '250.0', '750.0', '10000.0', '8000.0', '7000.0', '6000.0',
#                '4000.0', '3000.0', '1000.0', '1500.0', '2000.0', '11200.0', '500.0']


# boolean = X_train_columns.str.startswith(tuple(categories))
# indices = [*filter(boolean.get, boolean.index)]

# from pandas.api.types import CategoricalDtype
# for column in X_train.iloc[:, indices]:
#     X_train[column] = X_train[column].astype('category')
#     X_test[column] = X_test[column].astype('category')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Convert categories to categorical types"""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Imputation"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Define imputer for X_train, X_test set

import missingno as msno
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

n_neighbors_list = [3, 5, 7, 9]

imputer = KNNImputer(n_neighbors=1)

"""X_train imputation"""
X_train = pd.DataFrame(data = imputer.fit_transform(X_train), columns = X_train.columns, index = X_train.index)

"""X_test imputation"""
msno.heatmap(X_test)
X_test = pd.DataFrame(data = imputer.fit_transform(X_test), columns = X_test.columns, index = X_test.index)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Imputation"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Round values on X_train and X_test"""""""""""""""""""""""""""""""""""""""""""

# # Define a function to round all columns except those starting with "whoqol_"
# def round_columns(df):
#     return df.apply(lambda col: col.round(0) if not col.name.startswith('whoqol_') else col)

# # Apply the rounding function to both X_train and X_test
# X_train = round_columns(X_train)
# X_test = round_columns(X_test)

categories = ['participant_lang','q_a2_sex', 'q_a7_smoking_status', 'q_a8_father', 'q_a8_mother', 'q_a9_vertigo', 'q_a10_', 'q_a11_', 
              'q_a12_ext_sounds_prob', 'q_a13_hearing_difficulty', 'q_a14_', 'q_a15_', 'q_a16_', 'q_a17_tinnitus', 
              'q_b1_frequency', 'q_b2_day_pattern', 'q_b4_bother', 'q_b6_number_sounds', 'q_b7_onset_char', 'q_b9_',
              'q_b10_', 'q_b12_loudness_changes', 'q_b13_quality', 'q_b14_pitch', 'q_b15_localisation', 'q_b16_rhythmic',
              'q_b17_objective', 'q_b18_', 'q_b19_', 'q_b20_tinnitus_healthcare', 'q_b21_', 'mini_soises_', 'tinn_sever_',
               'b11_', 'b22_', 'b10_', 'a15_', 'Hearing Aid Indication', 'code_intervention_protocol', 'Age category', 'left_hearing_loss',
               'right_hearing_loss', 'tinnitus_matching', 'right_matching_type','left_matching_type', 'residual_inhibition', 
               'combined_residual_inhibition', 'left_residual_inhibition','right_residual_inhibition', '18000.0',
               '125.0', '250.0', '750.0', '10000.0', '8000.0', '7000.0', '6000.0',
               '4000.0', '3000.0', '1000.0', '1500.0', '2000.0', '11200.0', '500.0']

for category in categories:
    columns_to_round = [col for col in X_train.columns if col.startswith(category)]
    X_train[columns_to_round] = X_train[columns_to_round].round()
    X_test[columns_to_round] = X_test[columns_to_round].round()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Round values on X_train and X_test"""""""""""""""""""""""""""""""""""""""""""







"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Convert categories to categorical types after imputation"""""""""""""""""""""""""""""""""""""""""""
# convert to categorical dtype variables that are categorical
X_train_columns = pd.Series(X_train.columns)




boolean = X_train_columns.str.startswith(tuple(categories))
indices = [*filter(boolean.get, boolean.index)]

from pandas.api.types import CategoricalDtype
for column in X_train.iloc[:, indices]:
    X_train[column] = X_train[column].astype('category')
    X_test[column] = X_test[column].astype('category')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Convert categories to categorical types after imputation"""""""""""""""""""""""""""""""""""""""""""







"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Create Age category - Check relationship between age and thi score change and remove potential outliers for X_train,X_test"""""""""""""""""""""""""""""""""""""""""""

X_train['Age category'] = get_age_ranges(X_train['q_a1_age'])
X_test['Age category'] = get_age_ranges(X_test['q_a1_age'])

# Convert Age category to categorical variable
X_train['Age category'] = X_train['Age category'].astype('category')
X_test['Age category'] = X_test['Age category'].astype('category')

X_train.name = 'X_train'
X_test.name = 'X_test'

X_train, y_train = thi_diff_outliers(X_train, y_train, 'Age category')
X_test, y_test = thi_diff_outliers(X_test, y_test, 'Age category')


X_train = X_train.drop('q_a1_age', 1)
X_test = X_test.drop('q_a1_age', 1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Create Age category - Check relationship between treatment and thi score change and remove potential outliers for X_train,X_test"""""""""""""""""""""""""""""""""""""""""""








"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Normalize numerical features"""""""""""""""""""""""""""""""""""""""""""
# Normalize numerical features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = X_train.select_dtypes(include=[np.float])
numerical_cols_list = numerical_features.columns.tolist()
Fit = scaler.fit(X_train[numerical_cols_list])
X_train[numerical_cols_list] = Fit.transform(X_train[numerical_cols_list])
X_test[numerical_cols_list] = Fit.transform(X_test[numerical_cols_list])   

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Normalize numerical features"""""""""""""""""""""""""""""""""""""""""""








"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Recursive Feature Elimination"""""""""""""""""""""""""""""""""""""""""""
#feature selection with Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns


X_train_feat_select = X_train.astype('int')
X_train_feat_select = X_train_feat_select.values
y_train_feat_select = y_train.values
y_train_feat_select = y_train_feat_select.astype('int')


models = dict()
rfe_model = RandomForestRegressor()


#for no_features in range(14, 30):

rfe = RFE(rfe_model, n_features_to_select=17)

fit = Pipeline(steps=[('s',rfe),('m',rfe_model)])
fit = rfe.fit(X_train_feat_select, y_train_feat_select)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print("Feature included: %s" % fit.support_)

"""Drop columns from X_train and X_test after feature selection"""

feat_select_drop_list = [X_train.columns[col] for col in range(X_train_feat_select.shape[1]) if fit.ranking_[col] != 1]

X_train_temp = X_train.drop(feat_select_drop_list, axis=1)
X_test_temp = X_test.drop(feat_select_drop_list, axis=1)


# # RFECV
# selector = RFECV(rfe_model, step=1, cv=5)
# selector = selector.fit(X_train, y_train)

# print(selector.support_) # Selected features
# print(selector.ranking_) # Feature ranking

# selected_features = X_train.columns[selector.get_support()]

# X_train_temp = X_train[selected_features]
# X_test_temp = X_test[selected_features]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Recursive Feature Elimination"""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Normalize numerical features"""""""""""""""""""""""""""""""""""""""""""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = X_train_temp.select_dtypes(include=[np.float])
numerical_cols_list = numerical_features.columns.tolist()
Fit = scaler.fit(X_train_temp[numerical_cols_list])
X_train_temp[numerical_cols_list] = Fit.transform(X_train_temp[numerical_cols_list])
X_test_temp[numerical_cols_list] = Fit.transform(X_test_temp[numerical_cols_list]) 

    



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Normalize numerical features"""""""""""""""""""""""""""""""""""""""""""







"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Random Forest"""""""""""""""""""""""""""""""""""""""""""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

X_train_values = X_train_temp.values
X_test_values = X_test_temp.values
y_train_values = y_train.values
y_test_values = y_test.values


# Define the classifier
rfc = RandomForestRegressor()

# Define the hyperparameters to tune
hyperparameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use randomized search to find the best hyperparameters
random_search = RandomizedSearchCV(
    rfc,
    param_distributions=hyperparameters,
    n_iter=10,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

# Fit the model on the training data
random_search.fit(X_train_values, y_train_values)

# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)

# Use the best hyperparameters to fit the model on the training data
rfc_best = RandomForestRegressor(**random_search.best_params_)
rfc_best.fit(X_train_values, y_train_values)
    
y_pred_rand = rfc_best.predict(X_test_values) 
R2score_rand = r2_score(y_test, y_pred_rand)

rmse_rand = mean_squared_error(y_test, y_pred_rand, squared=False)

# Plot model
plt.scatter(y_test, y_pred_rand, c=np.abs(y_test - y_pred_rand), cmap='coolwarm')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.colorbar()
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values RF (R^2={:.2f})'.format(R2score_rand))
plt.show()

# Calculate variance of y target

variance = np.var(y_train_values)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Random Forest"""""""""""""""""""""""""""""""""""""""""""
   




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Lasso regression"""""""""""""""""""""""""""""""""""""""""""
from sklearn.linear_model import Lasso, Ridge
# Create the Lasso model
lasso = Lasso()

# Define the range of values for alpha
param_grid = {'alpha': np.logspace(-4, 4, 9)}

# Use GridSearchCV to tune the value of alpha
lasso_grid = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
lasso_grid.fit(X_train_values, y_train_values)

# Print the best value of alpha
print("Best value of alpha:", lasso_grid.best_params_['alpha'])

# Fit a Lasso regression model
lasso_reg = Lasso(alpha=lasso_grid.best_params_['alpha'])
lasso_reg.fit(X_train_values, y_train_values)

# Predict on the test set and calculate the mean squared error
lasso_predictions = lasso_reg.predict(X_test_values)
lasso_mse = mean_squared_error(y_test, lasso_predictions, squared=False)
lasso_r2 = r2_score(y_test, lasso_predictions)

plt.scatter(y_test, lasso_predictions, c=np.abs(y_test - lasso_predictions), cmap='coolwarm')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.colorbar()
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values Lasso (R^2={:.2f})'.format(lasso_r2))
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Lasso regression"""""""""""""""""""""""""""""""""""""""""""





"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Ridge regression"""""""""""""""""""""""""""""""""""""""""""


#Ridge
alphas = [0.1, 0.5, 1, 5, 10, 50, 100]

# Create a dictionary where the key is the parameter name and the value is the list of parameter values
param_grid = {'alpha': alphas}

# Create a Ridge regression model
ridge = Ridge()

# Create a GridSearchCV object
grid_search = GridSearchCV(ridge, param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_values, y_train_values)

# Print the best hyperparameters
print('Best alpha:', grid_search.best_params_['alpha'])

# Fit a Ridge regression model
ridge_reg = Ridge(alpha=grid_search.best_params_['alpha'])
ridge_reg.fit(X_train_values, y_train_values)

# Predict on the test set and calculate the mean squared error
ridge_predictions = ridge_reg.predict(X_test_values)
ridge_mse = mean_squared_error(y_test, ridge_predictions, squared=False)
ridge_r2 = r2_score(y_test, ridge_predictions)

plt.scatter(y_test, ridge_predictions, c=np.abs(y_test - ridge_predictions), cmap='coolwarm')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.colorbar()
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values Ridge (R^2={:.2f})'.format(ridge_r2))
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Ridge regression"""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Elastic Net"""""""""""""""""""""""""""""""""""""""""""

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV


regr = ElasticNetCV(cv=5, random_state=0)
regr.fit(X_train_values, y_train_values)

print("Best alpha: ", regr.alpha_)
print("Best l1_ratio: ", regr.l1_ratio_)


# Creating an instance of the ElasticNet model
elastic = ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_)

# Fitting the model to the training data
elastic.fit(X_train_values, y_train_values)

# Making predictions on the test data
elastic_pred = elastic.predict(X_test_values)

# Calculating the mean squared error of the predictions
elastic_mse = mean_squared_error(y_test, elastic_pred, squared=False)
elastic_r2 = r2_score(y_test, elastic_pred)

plt.scatter(y_test, elastic_pred, c=np.abs(y_test - elastic_pred), cmap='coolwarm')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.colorbar()
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values Elastic (R^2={:.2f})'.format(elastic_r2))
plt.show()
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Elastic Net"""""""""""""""""""""""""""""""""""""""""""           
            
            
#models[str(no_features)] = [R2score_rand, lasso_r2, ridge_r2, elastic_r2]




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Print Results"""""""""""""""""""""""""""""""""""""""""""   
print("Lasso MSE:", lasso_mse)
print("Lasso R2:", lasso_r2)
print("Ridge MSE:", ridge_mse)
print("Ridge R2:", ridge_r2)
print("RandomForest MSE:", rmse_rand)
print("RandomForest R2:", R2score_rand)
print("Elastic MSE", elastic_mse)
print("Elastic R2", elastic_r2)
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Print Results"""""""""""""""""""""""""""""""""""""""""""   






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Save model, scaler and features on disk"""""""""""""""""""""""""""""""""""""""""""  
#save the model to disk
# import pickle
# filename = 'flask/finalized_model_23.sav'
# pickle.dump(rfc_best, open(filename, 'wb'))

# # Save scaler to disk
# scaler_filename = './flask/scaler.pkl'
# pickle.dump(Fit, open(scaler_filename, 'wb'))


# # # Save X_train columns
# X_train_temp.to_pickle('flask/features.pkl')

# features_saved= pd.read_pickle('flask/features.pkl')

# features_saved_columns = features_saved.columns

# X_train_temp.columns

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Save model, scaler and features on disk"""""""""""""""""""""""""""""""""""""""""""  
