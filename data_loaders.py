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
    