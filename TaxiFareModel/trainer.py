from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,BaggingRegressor
from sklearn.svm import LinearSVR, NuSVR,SVR

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split,cross_val_score
from TaxiFareModel.data import clean_data, get_X_y,get_data

import joblib

from memoized_property import memoized_property

import mlflow
from mlflow.tracking import MlflowClient

from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y,experiment_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name=experiment_name

    def set_pipeline(self,model):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', model)
        ])
        return pipeline

    def cross_val_rmse(self,model):
        return cross_val_score(self.set_pipeline(model),self.X,self.y,n_jobs=-1,scoring='neg_root_mean_squared_error').mean()

    def run(self,model):
        """set and train the pipeline"""
        self.pipeline=self.set_pipeline(model)
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self,name):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self, f'{name}.joblib')
        pass


if __name__ == "__main__":
    model_dict={'linear':LinearRegression(),'ridge':Ridge(),'SGD':SGDRegressor(),'Ada':AdaBoostRegressor(),'GBR':GradientBoostingRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),'BaggingRegressor':BaggingRegressor(),
                'LinearSVR':LinearSVR(), 'NUSVR':NuSVR(),'SVR':SVR(),'XGB':XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)}

    for model in model_dict:
        df=get_data()
        cleaned_df=clean_data(df)
        X,y=get_X_y(cleaned_df)
        # X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)
        trainer=Trainer(X,y,"[FRANCE] [PARIS] [timfilhol] taxi_fare v2")

        rmse=trainer.cross_val_rmse(model_dict[model])

        trainer.save_model(model)

        trainer.mlflow_log_metric("rmse", rmse)
        trainer.mlflow_log_param("model", model)
        trainer.mlflow_log_param("student_name", 'Timothée Filhol')

        experiment_id = trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
