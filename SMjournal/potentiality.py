
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from .low_level import getFullCombination
from IPython.display import display,HTML



def importance_test(df1,y_col,weight_col=None,control_cols='not',const_cols='not'):
  df=df1.copy()
  if weight_col:
    df=df.loc[df[weight_col]>0]


  if const_cols=='not':
    const_cols=[i for i in df.columns if ('.hol' in i.lower()) or ('.mkt' in i.lower())]
  if control_cols=='not':
    control_cols=[i for i in df.columns if i not in const_cols+[y_col,weight_col,'Geographies','Weeks','non_lin_pred','lin_pred']]
  #
  weight=df[weight_col].to_numpy() if weight_col else None

  
  x_cols=control_cols+const_cols
  X_rand=np.ones([len(control_cols),len(control_cols)])-np.eye(len(control_cols))
  allBetas=[]
  y_miniReg=df[y_col].to_numpy()
  #meta
  allLosses=[]
  for i in tqdm(X_rand):
    # print(i)
    randomX_cols=getFullCombination(i,control_cols)
    df_x=pd.concat([df[randomX_cols],df[const_cols]

          ],axis=1)
    X_miniReg=df_x.to_numpy()

    #hago la regresion
    reg = LinearRegression().fit(X_miniReg, y_miniReg, weight)
    betas={df_x.columns[i]:reg.coef_[i] for i in range(len(df_x.columns))}
    allLosses.append(np.mean((reg.predict(X_miniReg)-y_miniReg)**2))
    allBetas.append(betas)
  #get base
  pre_y=df[y_col].to_numpy()
  df_x=pd.concat([df[x_cols],
          ],axis=1)
  pre_X=df_x.to_numpy()
  
  reg = LinearRegression().fit(pre_X, pre_y,weight)
  topred=pd.concat([df[x_cols],
          ],axis=1)
  topred_np=topred.to_numpy()
  df['lin_pred']=reg.predict(topred_np)
  baseError=np.mean((df['lin_pred'].to_numpy()-df[y_col].to_numpy())**2)
  ress=pd.DataFrame(dict(cols=control_cols, imp=np.array(allLosses)-baseError))
  plot0=ress.sort_values('imp').plot('cols', 'imp', 'barh')
  


  x_cols=control_cols+const_cols
  pre_y=df[y_col].to_numpy()
  df_x=pd.concat([df[x_cols],
          ],axis=1)
  pre_X=df_x.to_numpy()

  rf = RandomForestRegressor(1000, min_samples_leaf=5,n_jobs=-1)
  rf.fit(pre_X, pre_y);
  
  linn = LinearRegression().fit(pre_X, pre_y,weight)

  topred=pd.concat([df[x_cols],],axis=1)
  topred_np=topred.to_numpy()
  df['non_lin_pred']=rf.predict(topred_np)
  df['lin_pred']=linn.predict(topred_np)

  fulltotal=df.groupby(['Weeks']).mean()
  fulltotal=fulltotal.reset_index()
  fig=px.line(fulltotal,x='Weeks',y=[y_col,'non_lin_pred','lin_pred'])
  fig.show()
  #guardar_fig(fig,"bg overfitteado con vars de covid")
  display(HTML('<h1>First Linear, Second Non-Linear<h1>'))
  np.mean((df['non_lin_pred'].to_numpy()-df[y_col].to_numpy())**2)
  ress=pd.DataFrame(dict(cols=x_cols, imp=rf.feature_importances_))
  # ress.loc[ress["imp"]>0].plot('cols', 'imp', 'barh')
  plot1=ress.sort_values('imp').loc[ress["imp"]>0.001].plot('cols', 'imp', 'barh')
  return fig, plot0, plot1