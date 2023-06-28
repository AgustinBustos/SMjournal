import pyspssio
import random
import clevercsv
import pyreadstat
import pandas as pd
from google.colab import drive
from IPython.display import display

def input_path(message=""):  #translate copy path of windows
  raw_s=input(message)
  raw_s=raw_s.replace(r'"','')
  print("/content/drive/Shareddrives/"+'/'.join(raw_s.split("\\")[2:]))
  return "/content/drive/Shareddrives/"+'/'.join(raw_s.split("\\")[2:])
   
def get_data(path=False,sheet_name=0,usecols=None):  #easy get data from MES csv   ###############################################  path y sheet_name
    #import drive
    drive.mount("/content/drive")
    #transform str and get doc
    if path:
      newpath=path
    else:
      newpath=input_path('Input path:')
    
    if (newpath.split("/")[-1].split(".")[-1] == "xlsx") or (newpath.split("/")[-1].split(".")[-1] == "xls"):
        df=pd.read_excel(newpath,sheet_name=sheet_name)
    elif (newpath.split("/")[-1].split(".")[-1] == "csv"):
        df=pd.read_csv(newpath)
    elif (newpath.split("/")[-1].split(".")[-1] == "sav"):
        df=pd.read_spss(newpath,usecols=usecols)
        try:
           df=df.rename(columns={"WEEK": "Weeks", "MKT": "Geographies"})
           
        except:
           pass
    else:
        raise ValueError('Format Not Found')
    try:
      df=df.drop("Unnamed: 0",axis=1)
    except:
       pass
    try:
      df["Weeks"]=pd.to_datetime(df["Weeks"])
    except:
      pass
    
    return df
#"G:\Shared drives\Team members\Agustin\merk\00Q3\1websls\data.csv"
###########################################################################################################################################################


def to_sav(option='default',redundancy=False):  #try to read csv and parse it to sav
    #import drive
    drive.mount("/content/drive")
    newpath=input_path('Input path:')
    if (newpath.split("/")[-1].split(".")[-1] == "xlsx") or (newpath.split("/")[-1].split(".")[-1] == "xls"):
        df=pd.read_excel(newpath)
    elif (newpath.split("/")[-1].split(".")[-1] == "csv"):
        if option=='default':
          #df=pd.read_csv(newpath).reset_index()
          df = clevercsv.read_dataframe(newpath).reset_index()
          #df=df.T.reset_index().T
        elif option == 1:
          with open(newpath, "r", newline="") as fp:
            # you can use verbose=True to see what CleverCSV does
            dialect = clevercsv.Sniffer().sniff(fp.read(), verbose=False)
            fp.seek(0)
            reader = clevercsv.reader(fp, dialect)
            rows = list(reader)
          df=pd.DataFrame(rows)
          df.columns=["a"+str(i) for i in df.columns]
        elif option==2:
          df = clevercsv.read_dataframe(newpath)
        elif option==3:
          rows = clevercsv.read_table(newpath)
          df=pd.DataFrame(rows)  
    else:
        raise ValueError('Format Not Found')
    
    if redundancy:
      df=df.dropna(thresh=len(df)-round(0.99*round(len(df))), axis=1)
    display(df)
    newpath=input_path('Input folder to save:')
    randomo=random.randint(100000, 1000000)
    try:
      pyspssio.write_sav(newpath+f"/data_{randomo}.sav", df)
    except:
      pyreadstat.write_sav(df, newpath+f"/data_{randomo}.sav")
    print(f'Saved as data_{randomo}')
    print('Wait some minutes, the Google Drive is updating')
    return df

def fix_csv(option='default'): #try to fix csv
    #import drive
    drive.mount("/content/drive")
    newpath=input_path('Input path:')
    if (newpath.split("/")[-1].split(".")[-1] == "xlsx") or (newpath.split("/")[-1].split(".")[-1] == "xls"):
        df=pd.read_excel(newpath)
    elif (newpath.split("/")[-1].split(".")[-1] == "csv"):
        if option=='default':
          with open(newpath, "r", newline="") as fp:
            # you can use verbose=True to see what CleverCSV does
            dialect = clevercsv.Sniffer().sniff(fp.read(), verbose=False)
            fp.seek(0)
            reader = clevercsv.reader(fp, dialect)
            rows = list(reader)
          df=pd.DataFrame(rows)
        elif option==1:
          df = clevercsv.read_dataframe(newpath)
        elif option==2:
          rows = clevercsv.read_table(newpath)
          df=pd.DataFrame(rows)  
    else:
        raise ValueError('Format Not Found')
    
    df=df.dropna(thresh=len(df)-round(0.99*round(len(df))), axis=1)
    display(df)
    newpath=input_path('Input folder to save:')
    randomo=random.randint(100000, 1000000)
    df.to_csv(newpath+f"/data_{randomo}.csv",index=False)
    print(f'Saved as data_{randomo}')
    print('Wait some minutes, the Google Drive is updating')
