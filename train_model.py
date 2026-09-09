from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR=Path(__file__).resolve().parent
df=pd.read_csv(BASE_DIR/"data"/"graduate_admission.csv")
df=df.drop(columns=["Unnamed: 0","Serial No."],errors="ignore")
df.columns=[c.strip() for c in df.columns]
features=["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA","Research"]
target="Chance of Admit"
df[features+[target]]=df[features+[target]].apply(pd.to_numeric,errors="coerce")
df=df.dropna(subset=features+[target])
X_train,X_test,y_train,y_test=train_test_split(df[features],df[target],test_size=.20,random_state=42)

base=XGBRegressor(objective="reg:squarederror",random_state=42,n_jobs=-1)
params={"n_estimators":[100,200,300],"max_depth":[2,3,4],"learning_rate":[.02,.05,.10],"subsample":[.8,1.0],"colsample_bytree":[.8,1.0]}
search=GridSearchCV(base,params,cv=5,scoring="neg_mean_squared_error",n_jobs=-1)
search.fit(X_train,y_train)
model=search.best_estimator_
pred=model.predict(X_test)
mae=mean_absolute_error(y_test,pred); rmse=np.sqrt(mean_squared_error(y_test,pred)); r2=r2_score(y_test,pred)
joblib.dump(model,BASE_DIR/"models"/"admission_xgboost_model.joblib")
(BASE_DIR/"models"/"model_metadata.json").write_text(json.dumps({"features":features,"target":target,"best_params":search.best_params_,"metrics":{"MAE":float(mae),"RMSE":float(rmse),"R2":float(r2)}},indent=2))
print("MAE:",mae,"RMSE:",rmse,"R2:",r2)
