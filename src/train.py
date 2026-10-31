import json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_loader import ROOT, load_config, load_dataset
from src.preprocessing import make_preprocessor, prepare_xy

def main():
    cfg=load_config(); df=load_dataset(); X,y=prepare_xy(df,cfg)
    # Identical feature vectors (including any conflicting-label cases) stay
    # together, preventing repeated records from crossing evaluation folds.
    groups=pd.factorize(pd.MultiIndex.from_frame(X),sort=True)[0]
    split=StratifiedGroupKFold(n_splits=cfg["cv_folds"],shuffle=True,random_state=cfg["random_state"])
    train_idx,test_idx=next(split.split(X,y,groups=groups))
    Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx]
    ytr,yte=y.iloc[train_idx],y.iloc[test_idx]
    groups_tr,groups_te=groups[train_idx],groups[test_idx]
    assert set(groups_tr).isdisjoint(set(groups_te)), "A repeated feature vector crossed the train/test split."
    candidates={"svd":LinearDiscriminantAnalysis(solver="svd"),"lsqr_auto":LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")}
    cv=StratifiedGroupKFold(n_splits=cfg["cv_folds"],shuffle=True,random_state=cfg["random_state"])
    comparison=[]
    for name,model in candidates.items():
        pipe=Pipeline([("preprocess",make_preprocessor(Xtr)),("lda",model)])
        scores=cross_val_score(pipe,Xtr,ytr,groups=groups_tr,cv=cv,scoring="f1",n_jobs=1)
        comparison.append({"configuration":name,"cv_f1_mean":float(scores.mean()),"cv_f1_std":float(scores.std())})
    selected=max(comparison,key=lambda x:x["cv_f1_mean"])["configuration"]
    pipe=Pipeline([("preprocess",make_preprocessor(Xtr)),("lda",candidates[selected])])
    t=time.perf_counter(); pipe.fit(Xtr,ytr); training_seconds=time.perf_counter()-t
    t=time.perf_counter(); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]; batch=time.perf_counter()-t
    tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
    results={"dataset":{"rows":len(df),"columns":len(df.columns),"predictors":len(X.columns),"test_samples":len(yte),"class_counts":{"phishing":int(y.sum()),"legitimate":int((1-y).sum())}},
      "metrics":{"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"specificity":tn/(tn+fp),"roc_auc":roc_auc_score(yte,proba),"pr_auc":average_precision_score(yte,proba),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}},
      "timing":{"training_seconds":training_seconds,"test_inference_seconds":batch,"test_samples":len(yte),"mean_ms_per_sample":batch/len(yte)*1000,"throughput_samples_per_second":len(yte)/batch},
      "selection":{"candidates":comparison,"selected":selected,"cv":"5-fold stratified grouped training-only F1"},
      "split":{"method":"StratifiedGroupKFold","folds":cfg["cv_folds"],"random_state":cfg["random_state"],"group_key":"identical predictor vectors","train_samples":len(ytr),"test_samples":len(yte),"train_unique_groups":int(len(set(groups_tr))),"test_unique_groups":int(len(set(groups_te)))},
      "environment":{"python":platform.python_version(),"scikit_learn":sklearn.__version__,"platform":platform.platform()},"positive_class":"phishing (-1 mapped to 1)"}
    for p in ["artifacts","outputs","app/static/plots","models"]: (ROOT/p).mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump({"pipeline":pipe,"features":list(X.columns),"positive_label":-1},ROOT/"models/lda_pipeline.joblib")
    (ROOT/"outputs/results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    pd.DataFrame([results["metrics"]|results["timing"]]).to_csv(ROOT/"outputs/results.csv",index=False)
    pd.DataFrame(comparison).to_csv(ROOT/"outputs/lda_comparison.csv",index=False)
    plt.figure(figsize=(5,4)); y.value_counts().sort_index().rename(index={0:"Legitimate",1:"Phishing"}).plot.bar(); plt.ylabel("Samples"); plt.tight_layout(); plt.savefig(ROOT/"app/static/plots/class_distribution.png"); plt.close()
    plt.figure(figsize=(4,4)); plt.imshow([[tn,fp],[fn,tp]],cmap="Blues"); plt.xticks([0,1],["Legitimate","Phishing"]); plt.yticks([0,1],["Legitimate","Phishing"]); plt.xlabel("Predicted"); plt.ylabel("Actual");
    for i,v in enumerate([tn,fp,fn,tp]): plt.text(i%2,i//2,str(v),ha="center",va="center")
    plt.tight_layout(); plt.savefig(ROOT/"app/static/plots/confusion_matrix.png"); plt.close()
    fpr,tpr,_=roc_curve(yte,proba); plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.tight_layout(); plt.savefig(ROOT/"app/static/plots/roc_curve.png"); plt.close()
    prec,rec,_=precision_recall_curve(yte,proba); plt.figure(); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.savefig(ROOT/"app/static/plots/precision_recall.png"); plt.close()
    # Standardized-space coefficient magnitudes; useful as descriptive linear weights, not causal importance.
    names=pipe.named_steps["preprocess"].get_feature_names_out(); coefs=np.ravel(pipe.named_steps["lda"].coef_)
    pd.DataFrame({"feature":names,"coefficient":coefs}).assign(abs_coefficient=lambda d:d.coefficient.abs()).sort_values("abs_coefficient",ascending=False).to_csv(ROOT/"outputs/coefficients.csv",index=False)
    pd.DataFrame({"configuration":[x["configuration"] for x in comparison],"cv_f1_mean":[x["cv_f1_mean"] for x in comparison]}).plot.bar(x="configuration",y="cv_f1_mean",legend=False); plt.tight_layout(); plt.savefig(ROOT/"app/static/plots/lda_comparison.png"); plt.close()
    print(json.dumps(results,indent=2))
if __name__=="__main__": main()
