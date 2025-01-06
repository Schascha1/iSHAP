import pandas as pd
import numpy as np
# Datasets best suited for us are the next two

# Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico
# https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub
# Multiclass classification/ regression from 0-6

def column_to_numeric(series):
    values = series.unique()
    mapping = {}
    for ind,v in enumerate(values):
        mapping[v] = ind
    return mapping
    


def load_obesity():
    df = pd.read_csv("data/obesity/obesity.csv",sep=",")
    output = {}
    output["df"] = df.copy(True)
    output["target_name"] = "Weight Category"
    df["NObeyesdad"].replace({"Insufficient_Weight":0,"Normal_Weight":1,"Overweight_Level_I":2,"Overweight_Level_II":3,"Obesity_Type_I":4,"Obesity_Type_II":5,"Obesity_Type_III":6},inplace=True)
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        if col == "NObeyesdad":
            continue
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)   
    output["target"] = df["NObeyesdad"].to_numpy()
    df = df.drop("NObeyesdad",axis=1) 
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

# German Credit Dataset (Binary Classification Risk/No Risk)
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
def load_credit():
    df = pd.read_csv("data/credit/german_credit_data.csv",sep=",",index_col=0)
    df.dropna(inplace=True)
    output = {}
    df['Age'] = df['Age'].astype('int')
    df['Duration'] = df['Duration'].astype('int')
    df['Credit amount'] = df['Credit amount'].astype('int')
    df["Risk"].replace( {"good":"Low Risk","bad":"High Risk"},inplace=True)
    df["Job"].replace({0 : "unskilled non-resident" , 1 : "unskilled resident", 2 : "skilled", 3 : "highly skilled"},inplace=True)
    output["df"] = df.copy(True)
    df["Risk"].replace( {"Low Risk":0,"High Risk":1},inplace=True)
    output["target"] = df["Risk"].to_numpy()
    df = df.drop("Risk",axis=1) 
    output["mapper"] = {}
    output["target_name"] = "Risk of Default"
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)   
        index = df.columns.values.tolist().index(col)
        output["mapper"][index] = {v: k for k, v in replacement.items()}
    
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_default():
    df = pd.read_csv("data/default/default.csv",sep=",",index_col=0)
    df.dropna(inplace=True)
    for i in range(1,7):
        df.drop("PAY_{:d}".format(i),axis=1,inplace=True)
        df.drop("PAY_AMT{:d}".format(i),axis=1,inplace=True)
        df.drop("BILL_AMT{:d}".format(i),axis=1,inplace=True)
    output = {}
    df["CREDIT LIMIT"] = (df["CREDIT LIMIT"]/33).round()
    df["Default"].replace( {0:"No Default",1:"Default"},inplace=True)
    df["SEX"].replace({1:"male",2:"female"},inplace=True)
    df["EDUCATION"].replace({1:"graduate school",2:"university",3:"high school",4:"other"},inplace=True)
    df["MARRIAGE"].replace({1:"married",2:"single",3:"other"},inplace=True)
    output["df"] = df.copy(True)
    df["Default"].replace( {"No Default":0,"Default":1},inplace=True)
    output["target"] = df["Default"].to_numpy()
    df = df.drop("Default",axis=1) 
    output["mapper"] = {}
    output["target_name"] = "Risk of Default"
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)   
        index = df.columns.values.tolist().index(col)
        output["mapper"][index] = {v: k for k, v in replacement.items()}
    
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_wages():
    df = pd.read_csv("data/wages/wages.csv")
    output = {}
    df.drop(["ed","height"],axis=1,inplace=True)
    output["df"] = df.copy(True)
    for col in ["sex","race"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)
    output["target"] = df["earn"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("earn")
    output["target_name"] = "income"
    output["data"] = df.drop("earn",axis=1).to_numpy()

    return output


def load_student():
    df = pd.read_csv("data/student/student-por.csv",sep=";")
    df.dropna(inplace=True)
    output = {}
    df = df.drop(["G1","G2"],axis=1)
    output["df"] = df.copy(True)
    output["target"] = df["G3"].to_numpy()
    df = df.drop("G3",axis=1)
    output["target_name"] = "Grade"
    turn_to_numeric = list(filter(lambda x: not pd.api.types.is_numeric_dtype(df[x]),df.columns.values.tolist()))
    for col in turn_to_numeric:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)    
    output["data"] = df.to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    return output

def load_bike():
    df = pd.read_csv("data/bike/day.csv",index_col=0)
    df = df.drop(["instant","dteday","casual","registered","yr"],axis=1)

    df.dropna(inplace=True)
    output = {}
    output["df"] = df.copy(True)

    output["target"] = df["cnt"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("cnt")
    output["data"] = df.drop("cnt",axis=1).to_numpy()
    output["target_name"] = "Bike rentals"
    return output


def load_life():
    df = pd.read_csv("data/life-expectancy/life.csv")
    df.dropna(inplace=True)
    output = {}
    output["df"] = df.copy(True)
    for col in ["Country","Status"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)
    output["target"] = df["Life expectancy"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("Life expectancy")
    output["data"] = df.drop("Life expectancy",axis=1).to_numpy()
    output["target_name"] = "Life expectancy"
    return output

def load_adult():
    df = pd.read_csv("data/adult/adult.data")
    output = {}
    output["df"] = df.copy(True)
    df["class"].replace([" <=50K"," >50K"],[0,1],inplace=True)
    for col in ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)

    output["target"] = df["class"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("class")
    output["data"] = df.drop("class",axis=1).to_numpy()
    output["target_name"] = "Income >= 50k"

    return output

def load_insurance():
    df = pd.read_csv("data/insurance/insurance.csv")
    output = {}
    df['age'] = df['age'].astype('int')
    df['charges'] = df['charges'].astype('int')
    df['bmi'] = df['bmi'].astype('int')
    output["df"] = df.copy(True)
    for col in ["sex","smoker","region"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)

    output["target"] = df["charges"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("charges")
    output["data"] = df.drop("charges",axis=1).to_numpy()
    output["target_name"] = "Insurance Rate"
    return output
    
def load_boston():
    df = pd.read_csv("data/boston_housing/BostonHousing.csv")
    output = {}
    output["df"] = df.copy(True)
    output["target"] = df["medv"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("medv")
    output["data"] = df.drop("medv",axis=1).to_numpy()
    output["target_name"] = "Median value of owner-occupied homes in $1000's"
    return output

def load_covid():
    df = pd.read_csv("data/covid/covid.csv")
    output = {}
    df.drop(["outcome","id","patient_id","weekday_change_of_status","hour_change_of_status","weekday_admit","hour_admit","days_change_of_status","date_admit","date_change_of_status","hospital"],axis=1,inplace=True)
    df.drop(df[df["group"]=="Patient"].index,inplace=True)
    df.dropna(inplace=True)
    output["df"] = df.copy(True)
    df["group"].replace(["Expired","Discharged"],[0,1],inplace=True)
    for col in ["sex","race"]:
        replacement = column_to_numeric(df[col])
        df[col].replace(replacement,inplace=True)
    df.dropna(inplace=True)
    output["target"] = df["group"].to_numpy()
    output["feature_names"] = df.columns.values.tolist()
    output["feature_names"].remove("group")
    df.drop("group",axis=1,inplace=True)
    output["data"] = df.to_numpy()
    output["target_name"] = "group"
    return output