import pandas as pd
from src.preprocessing import prepare_xy, make_preprocessor

def test_target_mapping_and_identifier_exclusion():
    df=pd.DataFrame({'Index':[0,1], 'f':[1,2], 'class':[-1,1]})
    X,y=prepare_xy(df,{'target_column':'class','positive_label':-1,'identifier_columns':['Index']})
    assert list(X.columns)==['f'] and list(y)==[1,0]
    assert make_preprocessor(X).fit_transform(X).shape==(2,1)
