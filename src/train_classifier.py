"""Compare flight-disjoint classifiers and held-out permutation importance."""
import json, logging
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
FEATURES=['rcs_dbsm','micro_doppler_hz','radial_velocity_mps']
def train(returns, output):
    train_index,test_index=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=71).split(returns,groups=returns.flight_id))
    reference,heldout=returns.iloc[train_index],returns.iloc[test_index]
    assert not set(reference.flight_id)&set(heldout.flight_id)
    output.mkdir(parents=True,exist_ok=True); results={}
    for name,model in [('random_forest',RandomForestClassifier(n_estimators=120,random_state=71)),('gradient_boosting',GradientBoostingClassifier(random_state=71))]:
        model.fit(reference[FEATURES],reference.target_class); predicted=model.predict(heldout[FEATURES])
        importance=permutation_importance(model,heldout[FEATURES],heldout.target_class,n_repeats=8,random_state=71)
        results[name]={'classification':classification_report(heldout.target_class,predicted,output_dict=True,zero_division=0),'classes':list(model.classes_),'confusion':confusion_matrix(heldout.target_class,predicted,labels=model.classes_).tolist(),'permutation_importance':dict(zip(FEATURES,importance.importances_mean.tolist()))}
    (output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    logging.getLogger(__name__).info('Evaluated %d held-out flights',heldout.flight_id.nunique())
    return results
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO); train(pd.read_csv('data/returns.csv'),Path('artifacts'))
