import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from src.utils import save_objects
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path =os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(Self):
        Self.data_transformaton_config = DataTransformationConfig()

    def get_data_transformer_obj(self):    
        try:
            numerical_col =["writing_score","reading_score"]
            categorial_col=["gender",
            "race_ethnicity",
            "parental_level_of_education","lunch",
            "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(
                steps =[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    
                ]
            )

            logging.info("Numerical col standard scaler started")
            logging.info("categorial col encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_col),
                    ("cat_pipeline",cat_pipeline,categorial_col)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            logging.info("reading of train and test data")
            logging.info("obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_obj()
            target_col_name ="math_score"
            numerical_column=["writing_score","reading_score"] 

            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_df[target_col_name]

            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test_df[target_col_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and tesing dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            save_objects(
                file_path=self.data_transformaton_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformaton_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e,sys)





