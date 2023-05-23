# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:44:31 2021

@author: Konstantinos
"""
import numpy as np
import pandas as pd
from builtins import any as b_any
import scipy.stats as ss
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pointbiserialr 
import itertools
import matplotlib.pyplot as plt 
import seaborn as sns   

def calculate_thi_score(df):
    # Function to calculate thi score
    thi_categories = pd.DataFrame(columns = ['patient_id', 'thi_category'])
    thi_categories['patient_id'] = df['patient_id']
    # Calculate percentage of null on each row
    for row in range(len(df.index)):
        thi_score = np.round(df.iloc[row, 1:].sum())
        missing_percentage = df.iloc[row, 1:].isnull().sum().sum()*100/len(df.count(axis=0)) -1
        if missing_percentage <=20 and missing_percentage >0:
            missing_items = df.iloc[row, 1:].isnull().sum().sum()     # count number of missing values on a row
            thi_score = (np.round((thi_score/(25 - missing_items))*25)/2)*2
        elif missing_percentage >20:
            thi_score = np.nan
        
        # Temp calculation
        thi_categories['thi_category'].iloc[[row]] = thi_score
        
        # transform thi_score to each of the categories mentioned on data handling plan
    #     if not np.isnan(thi_score):            
    #         if 0 <= thi_score <= 16:
    #             thi_categories['thi_category'].iloc[[row]] = 0
    #         elif 18 <= thi_score <=36:
    #             thi_categories['thi_category'].iloc[[row]] = 1
    #         elif 38 <= thi_score <= 56:
    #             thi_categories['thi_category'].iloc[[row]] = 2
    #         elif 58 <= thi_score <= 76:
    #             thi_categories['thi_category'].iloc[[row]] = 3
    #         else:
    #             thi_categories['thi_category'].iloc[[row]] = 4
    #     else:
    #         thi_categories['thi_category'].iloc[[row]] = np.nan
            
    # thi_categories = thi_categories.astype('category')        
    return thi_categories


def calculate_minitq_score(df):
    # Function to calculate MiniTQ Score
    minitq_categories = pd.DataFrame(columns = ['patient_id', 'minitq_category'])
    minitq_categories['patient_id'] = df['patient_id']
    for row in range(len(df.index)):
        row_isnull = df.iloc[row, 1:].isnull().any()
        if row_isnull:
            minitq_categories['minitq_category'].iloc[[row]] = np.nan
        else:
            minitq_score = df.iloc[row, 1:].sum() 
            # Temp calculation    
            minitq_categories['minitq_category'].iloc[[row]] = minitq_score
            
            # if not np.isnan(minitq_score):
            #     # transform minitq_score to each of the categories mentioned on data handling plan
            #     if 1 <= minitq_score <= 7:
            #         minitq_categories['minitq_category'].iloc[[row]] = 0    # compensated, i.e. no clinically relevant impairment due to tinnitus
            #     elif 8< minitq_score <= 12:
            #         minitq_categories['minitq_category'].iloc[[row]] = 1    #  moderately affected
            #     elif 13 <= minitq_score <= 18:
            #         minitq_categories['minitq_category'].iloc[[row]] = 2    # severely affected
            #     else:
            #         minitq_categories['minitq_category'].iloc[[row]] = 3    #  extremely affected
                    
    minitq_categories = minitq_categories.astype('category')            
    return minitq_categories


def calculate_tfi_score(df):
    tfi_categories = pd.DataFrame(columns = ['patient_id', 'tfi_category', 'intrusive', 'sense of control', 'cognitive', 'sleep', 'auditory', 'relaxation', 'QOL', 'emotional'])
    tfi_categories['patient_id'] = df['patient_id']
    for row in range(len(df.index)):
        # Items #1 and #3 require a simple transformation from a percentage scale to a 0-10 scale
         question_1 = df.iloc[row, 1]
         question_3 = df.iloc[row, 3]
         if not np.isnan(question_1):
             df.iloc[row, 1] = question_1/10             
         if not np.isnan(question_3):
             df.iloc[row, 3] = question_3/10
             
         missing_percentage = df.iloc[row, 1:].isnull().sum().sum()*100/len(df.count(axis=0)) -1
         if missing_percentage <= 76:
             missing_items = df.iloc[row, 1:].isnull().sum().sum()
             valid_answers = 25 - missing_items
             tfi_score = (df.iloc[row, 1:].sum()/valid_answers)*10
             tfi_categories['tfi_category'].iloc[[row]] = tfi_score
         
        # Calculate subscale scores
        # Intrusive
        # subscales = ['intrusive', 'sense of control', 'cognitive', 'sleep', 'auditory', 'relaxation', 'QOL', 'emotional']
        
        # for subscale in subscales:
        #     if subscale == 'intrusive':
  
         intrusive = df.iloc[row, 1:4]
         missing_intrusive = intrusive.isnull().sum().sum()
         if missing_intrusive <= 1:
             valid_answers = 3 - missing_intrusive
             intrusive_score = (intrusive.sum()/valid_answers)*10
             tfi_categories['intrusive'].iloc[[row]] = intrusive_score
         
         sense = df.iloc[row, 4:7]
         missing_sense = sense.isnull().sum().sum()
         if missing_sense <= 1:
             valid_answers = 3 - missing_sense
             sense_score = (sense.sum()/valid_answers)*10
             tfi_categories['sense of control'].iloc[[row]] = sense_score
             
         cognitive = df.iloc[row, 7:10]
         missing_cognitive = cognitive.isnull().sum().sum()
         if missing_cognitive <= 1:
             valid_answers = 3 - missing_cognitive
             cognitive_score = (cognitive.sum()/valid_answers)*10
             tfi_categories['cognitive'].iloc[[row]] = cognitive_score
             
         sleep = df.iloc[row, 10:13]
         missing_sleep = sleep.isnull().sum().sum()
         if missing_sleep <= 1:
             valid_answers = 3 - missing_sleep
             sleep_score = (sleep.sum()/valid_answers)*10
             tfi_categories['sleep'].iloc[[row]] = sleep_score
             
         auditory = df.iloc[row, 13:16]
         missing_auditory = auditory.isnull().sum().sum()
         if missing_auditory <= 1:
             valid_answers = 3 - missing_auditory
             auditory_score = (auditory.sum()/valid_answers)*10
             tfi_categories['auditory'].iloc[[row]] = auditory_score
             
         relaxation = df.iloc[row, 16:19]
         missing_relaxation = relaxation.isnull().sum().sum()
         if missing_relaxation <= 1:
             valid_answers = 3 - missing_relaxation
             relaxation_score = (relaxation.sum()/valid_answers)*10
             tfi_categories['relaxation'].iloc[[row]] = relaxation_score
             
         qol = df.iloc[row, 19:23]
         missing_qol = qol.isnull().sum().sum()
         if missing_qol <= 1:
             valid_answers = 3 - missing_qol
             qol_score = (relaxation.sum()/valid_answers)*10
             tfi_categories['relaxation'].iloc[[row]] = relaxation_score
             
         emotional = df.iloc[row, 23:26]
         missing_emotional = emotional.isnull().sum().sum()
         if missing_emotional <= 1:
             valid_answers = 3 - missing_emotional
             emotional_score = (emotional.sum()/valid_answers)*10
             tfi_categories['emotional'].iloc[[row]] = emotional_score
             
         
    return tfi_categories
             
            
def calculate_whoqol_score(df):
      #Check if values are between 1-5, else NaN
     for column in df.iloc[:, 1:].columns:
         df.loc[~df[column].between(1,5), column] = np.nan
     # Reverse 3 negatively phrased items
     replace_list = ["question_3", "question_4", "question_26"]
     for col in replace_list:
         df[col] = df[col].map({1:5, 2:4, 3:3, 4:2, 5:1})
     
     whoqol_categories = pd.DataFrame(columns = ['patient_id', 'whoqol_dom1', 'whoqol_dom2', 'whoqol_dom3', 'whoqol_dom4'])
     
     # Delete cases with >20% missing data
     indexes_to_drop = []
     for row in range(len(df.index)):
        missing_items = df.iloc[row, 1:].isnull().sum()
        if missing_items >=21:
            indexes_to_drop.append(row)

     df.drop(df.index[indexes_to_drop], inplace=True)



     whoqol_categories['patient_id'] = df['patient_id']
            
     for row in range(len(df.index)):
         dom1 = df.iloc[row, [3, 4, 10, 15, 16, 17, 18]]
         missing_dom1 = dom1.isnull().sum()
         if missing_dom1 <=1:
             whoqol_categories.iloc[row, 1] = dom1.mean()*4
        
         dom2 = df.iloc[row, [5, 6, 7, 11, 19, 26]]
         missing_dom2 = dom2.isnull().sum()
         if missing_dom2 <=1:
             whoqol_categories.iloc[row, 2] = dom2.mean()*4
             
         dom3 = df.iloc[row, [20, 21, 22]]
         missing_dom3 = dom3.isnull().sum()
         if missing_dom3 <=1:
             whoqol_categories.iloc[row, 3] = dom3.mean()*4
             
         dom4 = df.iloc[row, [8, 9, 12, 13, 14, 23, 24, 25]]
         missing_dom4 = dom4.isnull().sum()
         if missing_dom4 <=2:
             whoqol_categories.iloc[row, 4] = dom4.mean()*4
            
     return whoqol_categories        
             
             
def calculate_phq9_score(df):
      
    phq9_categories = pd.DataFrame(columns=['patient_id', 'phq9_category'])
    phq9_categories['patient_id'] = df['patient_id']
      
    for row in range(len(df.index)):
        missing_items = df.iloc[row, 1:].isnull().any()
        if not missing_items:
            phq9_score = df.iloc[row, 1:].sum()
            if 1<= phq9_score <=4:
                phq9_categories['phq9_category'].iloc[[row]] = 0    # Minimal depression
            elif 5<= phq9_score <= 9:
                phq9_categories['phq9_category'].iloc[[row]] = 1    # Mild depression
            elif 10 <= phq9_score <= 14:
                phq9_categories['phq9_category'].iloc[[row]] = 2    # Moderate depression
            elif 15<= phq9_score <=19:
                phq9_categories['phq9_category'].iloc[[row]] = 3    # Moderately severe depression
            else:
                phq9_categories['phq9_category'].iloc[[row]] = 4    # Severe depression
                
    phq9_categories = phq9_categories.astype('category')      
    return phq9_categories       
             
             
             
def create_b10_categories(df):
    
    b_10 = df[['patient_id', 'q_b10_other1', 'q_b10_other2', 'q_b10_other3']].copy()

    b10_categories = pd.DataFrame(columns = ['patient_id', 'b10_BP/HF', 'b10_Thyroid', 'b10_Stomach', 'b10_Seizure_Anxiety/Sleep'])             
    b10_categories['patient_id'] = b_10['patient_id']  
    b10_categories = b10_categories.fillna(0)       
    
    bp_hf_listnames = ['Cholesterin Tabletten', 'Beta-Blocker', 'Belok zok', 'Ramipril', 'Simvastatin', 'Symvastatin',
                       'Candesartan', 'Candesattan', 'Folpick', 'bisoprolol', 'Bisoprolol', 'felodipine', 'Warfarin', 'Statins', 
                       'statin', 'Blood thinner', 'blood-thinning', 'tardisc', 'rapril', 'Antivastitin', 'Jodid100', 'Lercanidipin', 
                       'Losartan', 'Marcumar', 'Amlodipine', 'Amlodipin', 'Eliquis', 'Eplerenon', 'Entresto', 'Blutdrucksenker', 'Nomexor', 
                       'Ranexa', "Tioblis", "antihypertensive", "antihipertensivo", "Atacand", "atacan", "Altactone", "Pravastatin", "coversyl", 
                       "Αντιπηκτικη αγωγη", "Kopalia", "Exforge HCT"]
    
    thyroid_listnames = ['L-Thyroxin', "l Thyroxin", "L Thyroxin", "Levothyroxine", "levothyroxine", "Thyroxine", "Thyroxin", 
                         "Euthyrox", "Euthirox", "euthyrox", "Eutirox", "Thyrox", "Τ4", "Schilddrüsenhormone"]
    
    stomach_listnames = ["Lansoprazole", "lansoprazole", "Lanzoprazole", "Omeprazole", "Omeprazol", "omeprazol", "pantoprazole", 
                         "Lansaprozole", "Maaloxan", "Laprazol", "esomeprazol"]
    
    seiz_anx_sleep_listnames = ["zolpidem", "Duloxetin", "pregabalina", "Pregabalin", "Xanax", "DIAZEPAN", "Lorazepam", "valium", "gabapentina"]
    

    
    for row in range(len(b_10.index)):
        if b_10.iloc[row, 1:].isnull().sum() > 2 :
            continue
        else:
            for column in range(1, len(b_10.columns)):
                if b_10.iloc[row, column] is None:
                    continue
                elif b_any(x in b_10.iloc[row, column] for x in bp_hf_listnames):
                    b10_categories.iloc[row, 1] = 1
                elif b_any(x in b_10.iloc[row, column] for x in thyroid_listnames):
                    b10_categories.iloc[row, 2] = 1
                elif b_any(x in b_10.iloc[row, column] for x in stomach_listnames):
                    b10_categories.iloc[row, 3] = 1
                elif b_any(x in b_10.iloc[row, column] for x in seiz_anx_sleep_listnames):
                    b10_categories.iloc[row, 4] = 1   
                else:
                    continue
    b10_categories = b10_categories.astype('category')
    return b10_categories
             

def create_b22_categories(df):
    
    b_22 = df[['patient_id', 'q_b22_other1', 'q_b22_other2', 'q_b22_other3']].copy()

    b22_categories = pd.DataFrame(columns = ['patient_id', 'b22_Stress', 'b22_Sleep/Tiredness', 'b22_Depression', 'b22_Neck', 'b22_Noise/Music', 
                                             'b22_Silence', 'b22_Jaw/Teeth', 'b22_Alcohol', 'b22_Hearing Loss', 'b22_Exercise/Tension'])             
    b22_categories['patient_id'] = b_22['patient_id']  
    b22_categories = b22_categories.fillna(0)       
        
    stress_listnames = ["Angst", "Stres", "Anxiety", "anxiety","STRESS", "working conditions","ANXIETY","worried",
                        "stres","estrés","nsiedad","Άγχος","στρες","Στρεσογόνες","Στρες","Angstzustände","Sorgen","Belastung"]
    
    depression_listnames = ["epres"]
    
    exercise_tension_listnames = ["Verspan","training","movements","fizyczny","Exercise","tensi","hiking","Physical",
                                  "ένταση","κίνηση","Anspannung","Körperliche","Anstrengung"]
    
    sleep_tired_listnames = ["Schlaf","schlaf","Zmeczenie", "Brak", "niewyspanie","leep","ired", 
                             "nap","sueño","κούραση","ύπνου","Ärger","Burn"]
    
    noise_music_listnames = ["Geräusch","umgebungslärm ","Lärm","dzwieki","Halas","Musik","Nois","High frequency sounds","loud","Loud","nois",
                             "pitch","music","shouting","alarm","LOUD","NOISES","sounds","dzwieki","Sound","θορυβος","fuertes",
                             "Grundrauschen","μουσική","φασαρια","ruidosos","ηχο","ήχους","Straßengeräusche","Knall","Geräusche","Lautstärke","ruido"]
    
    neck_listnames = ["HWS","Atlas","Halswirbelsäule","Cervicales","Αυχενικο","Bechterew","Nacken","Bandscheibenvorfall",
                      "Jahr","szyi","Neck","neck"]  
    
    
    silence_listnames = ["Quietness","quiet","Cisza","Ciche","Silence","silencio","relax"]
    
    jaw_teeth_listnames = ["Kiefer","CMD","Bruxismo","Zähn","Zahn", "γνάθου","zahn","jaw","teeth"]
    
    alcohol_listnames = ["Alchohol","alcohol","Alcohol"]
    
    hearing_loss_listnames = ["Hörverlust","Hörsturz","hearing loss","Hearing loss"]
    
    for row in range(len(b_22.index)):
        if b_22.iloc[row, 1:].isnull().sum() > 2 :
            continue
        else:
            for column in range(1, len(b_22.columns)):
                if b_22.iloc[row, column] is None:
                    continue
                elif b_any(x in b_22.iloc[row, column] for x in stress_listnames):
                    b22_categories.iloc[row, 1] = 1
                elif b_any(x in b_22.iloc[row, column] for x in sleep_tired_listnames):
                    b22_categories.iloc[row, 2] = 1
                elif b_any(x in b_22.iloc[row, column] for x in depression_listnames):
                    b22_categories.iloc[row, 3] = 1
                elif b_any(x in b_22.iloc[row, column] for x in neck_listnames):
                    b22_categories.iloc[row, 4] = 1            
                elif b_any(x in b_22.iloc[row, column] for x in noise_music_listnames):
                    b22_categories.iloc[row, 5] = 1            
                elif b_any(x in b_22.iloc[row, column] for x in silence_listnames):
                    b22_categories.iloc[row, 6] = 1
                elif b_any(x in b_22.iloc[row, column] for x in jaw_teeth_listnames):
                    b22_categories.iloc[row, 7] = 1
                elif b_any(x in b_22.iloc[row, column] for x in alcohol_listnames):
                    b22_categories.iloc[row, 8] = 1
                elif b_any(x in b_22.iloc[row, column] for x in hearing_loss_listnames):
                    b22_categories.iloc[row, 9] = 1
                elif b_any(x in b_22.iloc[row, column] for x in exercise_tension_listnames):
                    b22_categories.iloc[row, 10] = 1
                else:
                    continue
    
    b22_categories = b22_categories.astype('category')
    return b22_categories


def create_a15_categories(df):
    
    a_15 = df[['patient_id', 'q_a15_other1', 'q_a15_other2', 'q_a15_other3']].copy()

    a15_categories = pd.DataFrame(columns = ['patient_id', 'a15_Back', 'a15_Shoulder', 'a15_Muscle Pain', 'a15_Jaw', 'a15_Joints', 'a15_Migranes', 'a15_Hip'])             
    a15_categories['patient_id'] = a_15['patient_id']  
    a15_categories = a15_categories.fillna(0)        
        
    back_listnames = ["Wirbelsäulen","Bandscheibenvorfälle","Back","back","Rücken","espalda","LWS","Hexenschuss","Lumbar","lumbar",
                      "Lendenwirbel"]
    
    shoulder_listnames = ["Schulter ","shoulder","łopatką","Shoulder","hombros"]
    
    musclepain_listnames = ["miesni","Fibromyalgia","fibromyalgia","Muskelschmerzen","Muskulatur","muskuläre","muscular"]
    
    jaw_listnames = ["jaw","Bruxismo","Kiefer","mandibular"]
    
    joint_listnames = ["ISG","Knie","Sehnen","Gelenk","Gout","joint","arthrit","Arthr", "κοκκαλα"]
    
    migranes_listnames = ["Migräne","Ημικρα","Migäne", "Migrena"]  
        
    hip_listnames = ["Ischias","Hüft","cadera","Lendenwirbelsäule"]
    
    
    for row in range(len(a_15.index)):
        if a_15.iloc[row, 1:].isnull().sum() > 2 :
            continue
        else:
            for column in range(1, len(a_15.columns)):
                if a_15.iloc[row, column] is None:
                    continue
                elif b_any(x in a_15.iloc[row, column] for x in back_listnames):
                    a15_categories.iloc[row, 1] = 1
                elif b_any(x in a_15.iloc[row, column] for x in shoulder_listnames):
                    a15_categories.iloc[row, 2] = 1
                elif b_any(x in a_15.iloc[row, column] for x in musclepain_listnames):
                    a15_categories.iloc[row, 3] = 1
                elif b_any(x in a_15.iloc[row, column] for x in jaw_listnames):
                    a15_categories.iloc[row, 4] = 1            
                elif b_any(x in a_15.iloc[row, column] for x in joint_listnames):
                    a15_categories.iloc[row, 5] = 1            
                elif b_any(x in a_15.iloc[row, column] for x in migranes_listnames):
                    a15_categories.iloc[row, 6] = 1
                elif b_any(x in a_15.iloc[row, column] for x in hip_listnames):
                    a15_categories.iloc[row, 7] = 1
                else:
                    continue
    
    a15_categories = a15_categories.astype('category')
    return a15_categories
    
    
    
def create_b11_categories(df):
  
  b_11 = df[['patient_id', 'q_b11_other1', 'q_b11_other2', 'q_b11_other3']].copy()

  b11_categories = pd.DataFrame(columns = ['patient_id', 'b11_Stress', 'b11_Sleep/Tiredness', 'b11_Depression', 'b11_Neck', 'b11_Noise/Music',
                                           'b11_Jaw', 'b11_Teeth', 'b11_Hearing Loss/Problems', 'b11_Οtitis', 'b11_Cold/Flu', 'b11_Ear Infection', 'b11_Head Trauma'])             
  b11_categories['patient_id'] = b_11['patient_id']  
  b11_categories = b11_categories.fillna(0)       

  
  stress_listnames = ["Stress","stress","Belastung","Cisnienie","Angststörung","Anxiety","Shock","anxiety","streß","Panikattacke",
                      "Nervios","STRESS","Sterss","estres","estrés","ansiedad","ΣΤΡΕΣ","Στρε","Umgebungsdruck","Sress","Zeitdruck"]
  
  depression_listnames = ["epression","Στενοχώρια"]
  
  
  sleep_tired_listnames =  ["Schlaf", "schlaf", "leep", "Overworked","Burn","burn","tiredness"]
  
  
  noise_music_listnames = ["Lärm","Explosion","halasie","muzyka","loud","Loud","music","anxiety","nois","explosions",
                           "LOUD","NOISE","gunfire","speakers","Nois","HEADPHONES","eadphone","acoustic shock","concert",
                           "Nachtclub","Alarm","Geräusch","ohne Gehörschutz","uido","κρότος","esplosión","ακουστικα μουσικης","sonido",
                           "Autobahn","Musik","Laute","Grundrauschen","Knalltrauma","laute","ήχο","Lautstärke"]
  
  neck_listnames = ["HWS","szyjnego","Nacken","cervical","cuello","Auffahrunfall",
                    "nacken","Halswirbel","CMD","neck", "Neck"]
  
  jaw_listnames = ["Kiefer","γναθου","KIEFER"]
  
  
  teeth_listnames = ["ruxismus","tooth","dental","Zahn","Zähn","teeth","Dental","Paradontose","Weisheitszahn","zahn"]
  
  hearing_loss_listnames = ["Hörprobleme","Hörsturz","Hörver","hearing loss","Hearing loss","Altersschwerhörigkeit","deafness",
                             "Hearing difficulties","Hörminderung","ακουστικης ικανοτητας","Gehörverlust","Gehörsturz",
                             "Hipoacusia","pérdida de audición","απώλεια ακοής","trauma acustico","Hearing Loss","Akustisches Trauma"]
  
  
  otitis_listnames = ["Zapalemie ucha srodkowego","Mittelohrentzündung","Dolor de oídos","Ohrenschmerzen","ΩΤΙΤΙΔΑ","Dolor de oido"]
  
  cold_flu_listnames = ["Erkältung","cold","Flu","flu","illness","asal","cold","ill","Heuschnupfen","Gripp","Resfriado"]
  
  ear_infection_listnames = ["ear infection","Zapalenie","Labyrinthitis","Ear infection","Infection","LABYRINTHITIS","Ohrenentzündung",
                             "ear inflammation","Ohrentzündungen","infecciones de oído"]
  
  head_trauma_listnames = ["Gehirnerschütterung","Head","Kopf"]
  
  
  for row in range(len(b_11.index)):
      if b_11.iloc[row, 1:].isnull().sum() > 2 :
          continue
      else:
          for column in range(1, len(b_11.columns)):
              if b_11.iloc[row, column] is None:
                  continue
              if b_any(x in b_11.iloc[row, column] for x in stress_listnames):
                  b11_categories.iloc[row, 1] = 1
              if b_any(x in b_11.iloc[row, column] for x in sleep_tired_listnames):
                  b11_categories.iloc[row, 2] = 1
              if b_any(x in b_11.iloc[row, column] for x in depression_listnames):
                  b11_categories.iloc[row, 3] = 1
              if b_any(x in b_11.iloc[row, column] for x in neck_listnames):
                  b11_categories.iloc[row, 4] = 1            
              if b_any(x in b_11.iloc[row, column] for x in noise_music_listnames):
                  b11_categories.iloc[row, 5] = 1            
              if b_any(x in b_11.iloc[row, column] for x in jaw_listnames):
                  b11_categories.iloc[row, 6] = 1
              if b_any(x in b_11.iloc[row, column] for x in teeth_listnames):
                  b11_categories.iloc[row, 7] = 1
              if b_any(x in b_11.iloc[row, column] for x in hearing_loss_listnames):
                  b11_categories.iloc[row, 8] = 1
              if b_any(x in b_11.iloc[row, column] for x in otitis_listnames):
                  b11_categories.iloc[row, 9] = 1
              if b_any(x in b_11.iloc[row, column] for x in cold_flu_listnames):
                  b11_categories.iloc[row, 10] = 1
              if b_any(x in b_11.iloc[row, column] for x in ear_infection_listnames):
                  b11_categories.iloc[row, 11] = 1
              if b_any(x in b_11.iloc[row, column] for x in head_trauma_listnames):
                  b11_categories.iloc[row, 12] = 1
              else:
                  continue
              
  b11_categories = b11_categories.astype('category')        
  return b11_categories  
  
  
    
 
def outliers_to_nan(df, features_list, range):
    df_columns = pd.Series(df.columns) # Get indexes of columns
    boolean = df_columns.str.startswith(tuple(features_list))
    indices = [*filter(boolean.get, boolean.index)]
     
    for column in df.iloc[:, indices]:        
        df.loc[~df[column].between(range[0],range[1]), column] = np.nan
        
    return df
    
    
    
def check_outlier_q_a3(q_a3):
    for row in range(len(q_a3.index)):   
        measurement_qa3 = q_a3[row]
        if measurement_qa3 < 130 or measurement_qa3 > 240:
            q_a3.replace(measurement_qa3, np.nan, inplace=True)
        else:
            continue
 
    return q_a3
    
def check_outlier_q_a4(q_a4):
    for row in range(len(q_a4.index)):   
        measurement_qa4 = q_a4[row]
        if measurement_qa4 < 35 or measurement_qa4 > 170:
            q_a4.replace(measurement_qa4, np.nan, inplace=True)
        else:
            continue
 
    return q_a4
    
    
def check_q_a6_alcohol(df):
    drinks_ml = [40, 125, 330]
    for row in range(len(df.index)):
        alcohol_val = df[row]
        if alcohol_val >= 40:
            for k in drinks_ml:
                if alcohol_val % k == 0:
                    df[row] = alcohol_val/k
                elif not any(alcohol_val % x == 0 for x in drinks_ml):
                    df[row] = np.nan
    return df
    
    
  
def corrX_new(corr_mtx, cut) :
       
    # Get correlation matrix and upper triagle
    corr_mtx = corr_mtx.abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))
    
    dropcols = list()
    
    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                 'v2.target','corr', 'drop' ]))
    
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else: 
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]
                
                s = pd.Series([ corr_mtx.index[row],
                up.columns[col],
                avg_corr[row],
                avg_corr[col],
                up.iloc[row,col],
                drop],
                index = res.columns)
        
                res = res.append(s, ignore_index = True)
    
    dropcols_names = calcDrop(res)
    
    return(dropcols_names)    
    
    
def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
    
    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))
     
    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop 
    poss_drop = list(set(poss_drop).difference(set(drop)))
    
    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        
    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)
         
    return drop    
    

def cramers_v_corr(df):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    corr_data = pd.DataFrame(columns=df.columns, index=df.columns)
    for i in range(0, len(df.columns)):
        for j in range(0, len(df.columns)):
            confusion_matrix = pd.crosstab(df[df.columns[i]], df[df.columns[j]])
        
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            correlation = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))    
            corr_data.iloc[i, j] = correlation
    
    return corr_data
    
    
    
    
def ordinal_corr(df):
    
    unique_values_df = pd.Series()
    drop_corr_ordinal = []
    for column in range(len(df.columns)):
        unique_values = len(df.iloc[:, column].unique())
        unique_values_df = unique_values_df.append(pd.Series([unique_values], index=[column]))
        
    unique_list = sorted(unique_values_df.unique().tolist())
    
    for cat in unique_list:
        idx = unique_values_df.index[unique_values_df == cat].tolist()
        if len(idx) == 1:
            continue
        else:
           dataframe_to_corr = df.iloc[:, idx]
           dataframe_to_corr.reset_index(drop=True, inplace=True)
           dataframe_to_corr = dataframe_to_corr.astype(float)
           corr_ordinal = dataframe_to_corr.corr(method='kendall')
           drop_corr_list = corrX_new(corr_ordinal, 0.3)
           if drop_corr_list:
               drop_corr_ordinal.append(drop_corr_list)

    return drop_corr_ordinal
    
    
def nominal_corr(df):
    
    corr_data = pd.DataFrame(columns=df.columns, index=df.columns)
    for i in range(0, len(df.columns)):
        for j in range(0, len(df.columns)):
            nominal_correlation = matthews_corrcoef(df.iloc[:, i], df.iloc[:, j])   
            corr_data.iloc[i, j] = nominal_correlation
    corr_data = corr_data.astype(float)

    return corr_data
      
    
def nom_num_corr(vec):
    nominal = vec[0]
    numeric = vec[1]
    corr_data = pd.DataFrame(columns= numeric.columns, index=nominal.columns)
    corr_pvalue = pd.DataFrame(columns= numeric.columns, index=nominal.columns)
    for i in range(0, len(nominal.columns)):
        for j in range(0, len(numeric.columns)):
            correlation = pointbiserialr(nominal.iloc[:, i], numeric.iloc[:, j])   
            corr_data.iloc[i, j] = correlation[0]
            corr_pvalue.iloc[i,j] = correlation[1]
    corr_data = corr_data.astype(float)
    corr_pvalue = corr_pvalue.astype(float)
    return corr_data, corr_pvalue 
    
    
    
def nom_num_corr_drop_list(corr_mtx, cut):
    # Same as drop_corr_new but modified to fulfil the criteria of the nominal & numerical correlation matrix
    corr_mtx = corr_mtx.abs()
    avg_corr_nom = corr_mtx.mean(axis = 1)
    avg_corr_num = corr_mtx.mean(axis = 0)
    
    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                 'v2.target','corr', 'drop' ]))
    
    for row in range(len(corr_mtx.index)):
        for col in range (len(corr_mtx.columns)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr_nom.iloc[row] > avg_corr_num.iloc[col]): 
                    drop = corr_mtx.index[row]
                else: 
                    drop = corr_mtx.columns[col]
                
                s = pd.Series([ corr_mtx.index[row],
                corr_mtx.columns[col],
                avg_corr_nom.iloc[row],
                avg_corr_num.iloc[col],
                corr_mtx.iloc[row, col],
                drop],
                index = res.columns)
        
                res = res.append(s, ignore_index = True)
    
    dropcols_names = calcDrop(res)    
    
    return(dropcols_names) 
    
    
# Use this function to check for difference in columns between X_train and X_test
# after one-hot encoding and add missing columns with zeros in order the two dataframes
# to have the same ammount of columns           
def onehot_diff_Xtrain_Xtest(vec):
    train_cat = vec[0]
    test_cat = vec[1]

    difference = train_cat.columns.difference(test_cat.columns)
    if difference.any():
        zeros_list = list(itertools.repeat(0, len(test_cat.index)))
        for diff_colname in difference:
            test_cat[diff_colname] = zeros_list
    
    return test_cat
    


# Function that changes treatment's code label to reccomended label e.g uniti-g --> G 
def fix_treatment_codes(df):
    capitalist = ['-A', '-B', '-C', '-D', '-E', '-F', '-G', '-H', '-I', '-J']
    lower_list = list((map(lambda x: x.lower(), capitalist)))
    capitalist.extend(lower_list)
    treat_col = df['code_intervention_protocol']  #df.iloc[:, -1]
    boolean = treat_col.str.endswith(tuple(capitalist))
    indices = [*filter(boolean.get, boolean.index)]     
    for element in indices:
        treat_label = df.loc[df.index[element], 'code_intervention_protocol']
        if  not treat_col.str.contains(treat_label).any():
            continue
        elif treat_label[-1].isupper():
             df.replace(treat_label, treat_label[-1], inplace=True)
        else:
             df.replace(treat_label, treat_label[-1].upper(), inplace=True)             

    return df


def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return



def get_age_ranges(age_col):
    age_category = []
    for age in age_col:
        if age >= 18 and age < 28:
            age_category.append(0)
        elif age >= 28 and age < 38:
            age_category.append(1)
        elif age >= 38 and age < 48:
            age_category.append(2)
        elif age >= 48 and age < 58:
            age_category.append(3)
        elif age >= 58 and age < 68:
            age_category.append(4)
        else:
            age_category.append(5)
  
    return age_category



# Preprocess left and right matching loudness as well as right and left masking level
def audiological_loudness_and_masking(df, left_column, right_column):
    for i, row in df.iterrows():
        
        # check if tinnitus_matching value is 0
        if row.tinnitus_matching == 0:
            df.at[i, left_column] = 0
            
            # check if right_matching_loudness value is 999 or 888
            if row[right_column] in [999, 888]:
                df.at[i, right_column] = np.nan
                
        # check if tinnitus_matching value is 1
        elif row.tinnitus_matching == 1:
            df.at[i, right_column] = 0
            
            # check if left_matching_loudness value is 999 or 888
            if row[left_column] in [999, 888]:
                df.at[i, left_column] = np.nan
                
        # check if tinnitus_matching value is 2
        elif row.tinnitus_matching == 2:
            
            # check if both left and right matching loudness are 999 or 888
            if (row[left_column] in [999, 888]) and (row[right_column] in [999, 888]):
                df.at[i, left_column] = np.nan
                df.at[i, right_column] = np.nan
            
            # check if only right_matching_loudness value is 999 or 888
            elif row[right_column] in [999, 888]:
                df.at[i, right_column] = row[left_column]
            
            # check if only left_matching_loudness value is 999 or 888
            elif row[left_column] in [999, 888]:
                df.at[i, left_column] = row[right_column]
        else:
            df.at[i, right_column] = np.nan
            df.at[i, left_column] = np.nan
    return df


# Check relationship between treatment or age and thi score change and remove potential outliers for X_train,X_test
def thi_diff_outliers(X, y, feature):
    
    X['THI_diff'] = X['thi_score_baseline'] - y
    merged = pd.concat([X, y], axis=1)
    sns.boxplot(x = feature, y = 'THI_diff', data=X, whis=1.5)
    plt.title(X.name)
    plt.show()
    
    # Drop potential outliers identified from the box plot
    q1 = X['THI_diff'].quantile(0.25)
    q3 = X['THI_diff'].quantile(0.75)
    iqr = q3 - q1
    merged = merged[(X['THI_diff'] >= q1 - 1*iqr) & (X['THI_diff'] <= q3 + 1*iqr)]
    X = merged.drop(columns=['thi_score_final'])
    y = merged['thi_score_final']
    
    
    # Drop THI_diff column from X_train, X_test
    X = X.drop('THI_diff', 1)
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y


            
             
             
             
             
             