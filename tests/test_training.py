import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from src.preprocessing import make_preprocessor

def test_lda_pipeline_fits():
    X=pd.DataFrame({'a':[0,1,2,3,4,5], 'b':[1,0,1,0,1,0]}); y=[0,0,0,1,1,1]
    pipe=Pipeline([('preprocess',make_preprocessor(X)),('lda',LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))])
    pipe.fit(X,y)
    assert len(pipe.predict(X))==6 and pipe.predict_proba(X).shape==(6,2)
