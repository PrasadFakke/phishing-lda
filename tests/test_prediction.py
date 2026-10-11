import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from src.predict import predict_record
from src.preprocessing import make_preprocessor

def test_valid_prediction_and_invalid_features():
    X=pd.DataFrame({'a':[0,1,2,3,4,5], 'b':[1,0,1,0,1,0]}); y=[0,0,0,1,1,1]
    p=Pipeline([('preprocess',make_preprocessor(X)),('lda',LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))]).fit(X,y)
    bundle={'pipeline':p,'features':['a','b']}
    r=predict_record({'a':0,'b':1},bundle)
    assert r['prediction'] in ('phishing','legitimate') and 0<=r['phishing_probability']<=1
    with pytest.raises(ValueError): predict_record({'a':1},bundle)
