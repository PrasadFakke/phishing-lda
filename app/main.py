from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
import pandas as pd
from src.data_loader import ROOT, load_config, load_dataset
from src.preprocessing import prepare_xy
from src.predict import predict_record

app=FastAPI(title="LDA Phishing Website Classifier")
app.mount("/static",StaticFiles(directory=ROOT/"app/static"),name="static")
templates=Jinja2Templates(directory=ROOT/"app/templates")

@app.get("/api/dataset")
def dataset_info():
    df=load_dataset(); cfg=load_config(); X,y=prepare_xy(df,cfg)
    return {"rows":len(df),"columns":len(df.columns),"features":list(X.columns),"dtypes":{c:str(df[c].dtype) for c in X.columns},"missing":df.isna().sum().to_dict(),"class_counts":{"phishing":int(y.sum()),"legitimate":int((1-y).sum())},"preview":df.head(8).to_dict(orient="records")}

@app.get("/api/row/{idx}")
def dataset_row(idx:int):
    X,_=prepare_xy(load_dataset(),load_config())
    if idx<0 or idx>=len(X): raise HTTPException(404,"Dataset row is out of range.")
    return X.iloc[idx].to_dict()

@app.get("/",response_class=HTMLResponse)
@app.get("/{page}",response_class=HTMLResponse)
def page(request:Request,page:str="overview"):
    allowed={"overview","dataset","methodology","evaluation","predict"}
    if page not in allowed: raise HTTPException(404,"Page not found")
    data={"request":request,"page":page,"title":page.title()}
    data["model_ready"]=(ROOT/"models/lda_pipeline.joblib").exists() and (ROOT/"outputs/results.json").exists()
    if page=="dataset": data["dataset"]=dataset_info()
    if page=="evaluation":
        p=ROOT/"outputs/results.json"; data["results"]=json.loads(p.read_text()) if p.exists() else None
    if page=="predict":
        df=load_dataset(); X,_=prepare_xy(df,load_config()); data["rows"]=[{"id":int(i),"values":X.iloc[i].to_dict()} for i in range(min(50,len(X)))]
    return templates.TemplateResponse(request=request,name=f"{page}.html",context=data)

@app.post("/api/predict")
async def predict(request:Request):
    try:
        payload=await request.json()
        if "row" in payload:
            df=load_dataset(); X,_=prepare_xy(df,load_config()); idx=int(payload["row"])
            if idx<0 or idx>=len(X): raise ValueError("Dataset row is out of range.")
            record=X.iloc[idx].to_dict()
        else: record=payload
        return predict_record(record)
    except FileNotFoundError as e: raise HTTPException(503,str(e))
    except Exception as e: raise HTTPException(400,str(e))
