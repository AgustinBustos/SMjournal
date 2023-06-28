import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
from IPython.display import display


from .low_level import getFullCombination, getVIFs, getIndexOfCol3



#############################################################################################################################################################
#second page


def vif_distance(df,tounderstand,stochQ=1000):   #transform meta vif analisys to block modeling
  #'searchcont
  X_rand=np.random.randint(2, size=[stochQ,len(tounderstand)])
  X_rand=X_rand[np.where(X_rand.sum(axis=1)>=2)]
  allVIFs=[]

  for i in tqdm(X_rand):
    # print(i)
    randomX_cols=getFullCombination(i,tounderstand)
    allVIFs.append(getVIFs(randomX_cols,df))

  for i in allVIFs:
    for j in i.keys():
      if i[j]>3000:
        i[j]=3000
  
  interest=tounderstand

  impacten=tounderstand

  fullTest=tounderstand
  matoPato=[]
  for k in fullTest:
    var=k#x_cols[ind]
    metaY=np.array([allVIFs[i][var] for i in getIndexOfCol3(var,allVIFs)])
    metaX=X_rand[getIndexOfCol3(var,allVIFs)]
    metaReg = LinearRegression(fit_intercept=False).fit(metaX, metaY)
    # meta_betas
    normed_coef=(metaReg.coef_-metaReg.coef_.mean())/metaReg.coef_.std()
    normed_coef=metaReg.coef_
    matoPato.append(list(normed_coef))
  duff=pd.DataFrame(np.array(matoPato).transpose())
  duff.columns=fullTest
  duff.index=impacten

  duff.values[[np.arange(duff.shape[0])]*2] = 0

  to_test=interest

  to_test_index=impacten

  filtered_duff=duff.loc[to_test_index,to_test]
  simil=np.minimum(filtered_duff.to_numpy(),filtered_duff.to_numpy().T)

  simil[simil < 0] = 0
  disimil=1/simil
  disimil[disimil > 200] = 200
  disimil=np.log(disimil+1)
  n = disimil.shape[0]
  disimil[range(n), range(n)] = 0
  disimil=disimil/np.max(disimil)

  simil=1-disimil
  similitude=pd.DataFrame(simil)
  similitude.columns =tounderstand
  similitude.index=tounderstand
  return similitude



def collinearity_test(df,tounderstand=None,distance="vif"):   #full test of collinearity
  tounderstand=tounderstand if tounderstand else df.columns[2:] 
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

  #block distance
  #correlations
  correlations=df[to_test].corr().fillna(0)
  if distance=="corr":
    
    similarity=abs(correlations)
  elif distance=="vif":
    similarity=vif_distance(df,tounderstand,stochQ=1000)
  

  plt.figure(figsize=(12,5))
  dissimilarity = 1 - similarity
  Z = linkage(squareform(dissimilarity), 'complete')

  dendrogram(Z, labels=to_test, orientation='top', 
            leaf_rotation=90);
  plt.tight_layout()
  plt.savefig('maybe.svg', format='svg', dpi=1200)

  ########third
  px.line(((df.groupby('Weeks').mean()-df.groupby('Weeks').mean().mean())/df.groupby('Weeks').mean().std()).reset_index(),x='Weeks',y=tounderstand).show()


