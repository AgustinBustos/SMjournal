
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from .low_level import getFullCombination

def importance_test(df,control_cols,const_cols,y_col,weight_col,type="non lin"):
  if type=="lin":
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
    weight=df[weight_col].to_numpy()
    reg = LinearRegression().fit(pre_X, pre_y,weight)
    topred=pd.concat([df[x_cols],
            ],axis=1)
    topred_np=topred.to_numpy()
    df['pre_yhat']=reg.predict(topred_np)
    baseError=np.mean((df['pre_yhat'].to_numpy()-df[y_col].to_numpy())**2)
    ress=pd.DataFrame(dict(cols=control_cols, imp=np.array(allLosses)-baseError))
    ress.sort_values('imp').plot('cols', 'imp', 'barh')
  elif type=="non lin":
    x_cols=control_cols+const_cols
    pre_y=df[y_col].to_numpy()
    df_x=pd.concat([df[x_cols],
            ],axis=1)
    pre_X=df_x.to_numpy()

    rf = RandomForestRegressor(100, min_samples_leaf=10,n_jobs=-1)
    rf.fit(pre_X, pre_y);

    topred=pd.concat([df[x_cols],],axis=1)
    topred_np=topred.to_numpy()
    df['pre_yhat']=rf.predict(topred_np)

    fulltotal=df.groupby(['Weeks']).mean()
    fulltotal=fulltotal.reset_index()
    fig=px.line(fulltotal,x='Weeks',y=[y_col,'pre_yhat'])
    fig.show()
    #guardar_fig(fig,"bg overfitteado con vars de covid")

    np.mean((df['pre_yhat'].to_numpy()-df[y_col].to_numpy())**2)
    ress=pd.DataFrame(dict(cols=x_cols, imp=rf.feature_importances_))
    # ress.loc[ress["imp"]>0].plot('cols', 'imp', 'barh')
    ress.sort_values('imp').loc[ress["imp"]>0.001].plot('cols', 'imp', 'barh')