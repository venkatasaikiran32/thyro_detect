import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
import os
from src.constant import *
from src.utils.main_utils import MainUtils 


@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(artifact_folder, 'data_transformation')
    transformed_train_file_path = os.path.join(data_transformation_dir, 'train.csv')
    transformed_test_file_path = os.path.join(data_transformation_dir, 'test.csv')
    transformed_object_file_path = os.path.join(data_transformation_dir, 'preprocessing.pkl')
    

class DataTransformation:
    def __init__(self,valid_data_dir):
        self.valid_data_dir = valid_data_dir
        self.data_transformation_config=DataTransformationConfig()
        self.utils=MainUtils()
    

    @staticmethod
    def get_merged_batch_data(valid_data_dir: str) -> pd.DataFrame:
        """
        Method Name :   get_merged_batch_data
        Description :   This method reads all the validated raw data from the valid_data_dir and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            raw_files = os.listdir(valid_data_dir)
            csv_data = []
            for filename in raw_files:
                data = pd.read_csv(os.path.join(valid_data_dir, filename))
                csv_data.append(data)

            merged_data = pd.concat(csv_data)


            return merged_data
        except Exception as e:
            raise CustomException(e, sys)
    

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            #a function that returns the dataframe for that corresponding file
            df=self.get_merged_batch_data(valid_data_dir=self.valid_data_dir)
            categorical_features, continuous_features, discrete_features=self.utils.identify_feature_types(df)

            num_discrete_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mode")),
                ("scaler",StandardScaler())

                ]
            )

            num_continuous_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
               
                 ("scaler",StandardScaler())
                ]

            )

            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),

                #handle_unknown='ignore'--> it will also handles the categorical data even it is not present in training data
                #if we dont use the param then it may cause error during transform on test data .

                 ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                 ("scaler",StandardScaler(with_mean=False)) 
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical discrete  columns: {discrete_features}")
            logging.info(f"Numerical continuous columns: {continuous_features}")


            preprocessor=ColumnTransformer(
                [
                ("num_discrete_pipeline",num_discrete_pipeline,discrete_features),
                ("num_continuous_pipeline",num_continuous_pipeline,continuous_features),
                ('cat_pipeline',cat_pipeline,categorical_features)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            dataframe = self.get_merged_batch_data(valid_data_dir=self.valid_data_dir)
            dataframe = self.utils.remove_unwanted_spaces(dataframe)
            dataframe.replace('?', np.NaN, inplace=True)  # replacing '?' with NaN values for imputation

            sampler = SMOTE(random_state=30)
            X = dataframe.drop(columns=[TARGET_COLUMN])
            y = np.array(dataframe[TARGET_COLUMN])
            x_sampled, y_sampled = sampler.fit_resample(X, y)

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # Split the data
            x_train, x_test, y_train, y_test = train_test_split(
            x_sampled, y_sampled, test_size=0.3, random_state=30
        )

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Transform the data
            x_train_scaled = preprocessing_obj.fit_transform(x_train)
            x_test_scaled = preprocessing_obj.transform(x_test)

            # Save the preprocessing object
            logging.info("Saving preprocessing object.")
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessing_obj)

        # Save the transformed datasets
            logging.info('Storing test and train data')
            train_df = pd.DataFrame(np.c_[x_train_scaled, y_train])
            test_df = pd.DataFrame(np.c_[x_test_scaled, y_test])

            train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path, index=False)


            #modifying y_train and y_test
            #our y(target column) values starting from 1,2,3 but the scikit-learn wants the values should be zero-indexed
            #that means values of target label should start from 0,1,2,..,
            y_train=y_train-1
            y_test=y_test-1

            return x_train_scaled, y_train, x_test_scaled, y_test, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
