import pandas as pd
from google.colab import drive
def input_path(message):
  raw_s=input(message)
  raw_s=raw_s.replace(r'"','')
  return "/content/drive/Shareddrives/"+'/'.join(raw_s.split("\\")[2:])
   
def get_data():
    #import drive
    drive.mount("/content/drive")
    #transform str and get doc
    newpath=input_path('Input path:')
    if (newpath.split("/")[-1].split(".")[-1] == "xlsx") or (newpath.split("/")[-1].split(".")[-1] == "xls"):
        df=pd.read_excel(newpath)
    elif (newpath.split("/")[-1].split(".")[-1] == "csv"):
        df=pd.read_csv(newpath)
    else:
        raise ValueError('Format Not Found')
    try:
      df=df.drop("Unnamed: 0",axis=1)
      df["Weeks"]=pd.to_datetime(df["Weeks"])
    except:
       pass
    
    return df
#"G:\Shared drives\Team members\Agustin\merk\00Q3\1websls\data.csv"
###########################################################################################################################################################

import pyspssio
import random
import clevercsv
import pyreadstat
def to_sav(option='default'):
    #import drive
    
    newpath=input_path('Input path:')
    if (newpath.split("/")[-1].split(".")[-1] == "xlsx") or (newpath.split("/")[-1].split(".")[-1] == "xls"):
        df=pd.read_excel(newpath)
    elif (newpath.split("/")[-1].split(".")[-1] == "csv"):
        if option=='default':
          df=pd.read_csv(newpath)
        elif option == 1:
          with open(newpath, "r", newline="") as fp:
            # you can use verbose=True to see what CleverCSV does
            dialect = clevercsv.Sniffer().sniff(fp.read(), verbose=False)
            fp.seek(0)
            reader = clevercsv.reader(fp, dialect)
            rows = list(reader)
          df=pd.DataFrame(rows)
        elif option==2:
          df = clevercsv.read_dataframe(newpath)
        elif option==3:
          rows = clevercsv.read_table(newpath)
          df=pd.DataFrame(rows)  
    else:
        raise ValueError('Format Not Found')
    
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

def fix_csv(option='default'):
    #import drive
    
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


#############################################################################################################################################################
#second page
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
def adjRsquare(pre_y,r,vars):
  l=len(pre_y)
  return 1 - (1-r2_score(pre_y, r)) * (l-1)/(l-len(vars)-1)
def getVIFs(vars,df):
  result={}
  for i in vars:
    pre_y=df[i].to_numpy()
    df_x=pd.concat([df[[j for j in vars if j!=i]],
                  
            ],axis=1)
    pre_X=df_x.to_numpy()
    # weight=df[weight_col].to_numpy()

    #hago la regresion
    reg = LinearRegression().fit(pre_X, pre_y)   #############OJOTA ,weight
    #reg = LinearRegression().fit(pre_X, pre_y)
    result[i]=1/(1-r2_score(pre_y, reg.predict(pre_X)) )
    result[i]=1/(1-adjRsquare(pre_y, reg.predict(pre_X),vars) )
  return result
def collinearity_test(df,tounderstand):
  to_test=tounderstand
  display(getVIFs(tounderstand,df))
  # set figure size
  correlation=df[to_test].corr().fillna(0)
  filter=0
  corr_numpy=correlation.to_numpy()
  indices=corr_numpy.argsort()[:,-2]
  corr_numpy
  finalIndices=[]
  for i in range(len(corr_numpy)):
    value=corr_numpy[i,indices[i]]
    if value>filter or value<-filter:
      finalIndices.append(i)
  finalIndices
  finalCorr=correlation.iloc[finalIndices,finalIndices]

  plt.figure(figsize=(10,7))

  # Generate a mask to onlyshow the bottom triangle
  mask = np.triu(np.ones_like(finalCorr, dtype=bool))

  # generate heatmap
  sns.heatmap(finalCorr, annot=True, mask=mask, vmin=-1, vmax=1,cmap="rocket_r")
  plt.title('Correlation Coefficient Of Predictors')
  plt.show()


  
  data=df[to_test]
  #correlations
  correlations=df[to_test].corr().fillna(0)
  similarity=abs(correlations)

  similarity
  

  plt.figure(figsize=(12,5))
  dissimilarity = 1 - similarity
  Z = linkage(squareform(dissimilarity), 'complete')

  dendrogram(Z, labels=to_test, orientation='top', 
            leaf_rotation=90);
  plt.tight_layout()
  plt.savefig('maybe.svg', format='svg', dpi=1200)

  ########third
  px.line(((df.groupby('Weeks').mean()-df.groupby('Weeks').mean().mean())/df.groupby('Weeks').mean().std()).reset_index(),x='Weeks',y=tounderstand).show()

