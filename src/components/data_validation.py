import re
import os
import shutil
import sys
import json
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass
from src.constant import *

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(artifact_folder, 'data_validation')
    valid_data_dir: str = os.path.join(data_validation_dir, 'validated')
    invalid_data_dir: str = os.path.join(data_validation_dir, 'invalid')
    schema_config_file_path: str = os.path.join('config', 'training_schema.json')


class DataValidation:
    def __init__(self, raw_data_store_dir: Path):
        logging.info('Entered into DataValidation class')
        self.raw_data_store_dir = raw_data_store_dir
        self.data_validation_config = DataValidationConfig()
        self.utils = MainUtils()
        logging.info('Exited from DataValidation class')

    def valuesFromSchema(self):
        try:
            logging.info('Entered into valuesFromSchema')
            with open(self.data_validation_config.schema_config_file_path, 'r') as f:
                dic = json.load(f)
            length_of_date_stamp_in_file = dic['LengthOfDateStampInFile']
            length_of_time_stamp_in_file = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']
            return length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns

        except Exception as e:
            raise CustomException(e, sys)

    def validate_file_name(self, file_path: str) -> bool:
        try:
            logging.info('Entered into validate_file_name')
            file_name = os.path.basename(file_path)
            regex = r"^[a-zA-Z0-9_]+\.csv$"  # Regex for filenames in the format filename.csv

            return bool(re.match(regex, file_name))

        except Exception as e:
            raise CustomException(e, sys)

    def validate_no_of_columns(self, file_path: str, schema_no_of_columns: int) -> bool:
        try:
            logging.info("Entered into validate_no_of_columns")
            dataframe = pd.read_csv(file_path)
            return len(dataframe.columns) == schema_no_of_columns

        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values_in_whole_column(self, file_path: str) -> bool:
        try:
            logging.info("Entered into validate_missing_values_in_whole_column")
            dataframe = pd.read_csv(file_path)
            no_of_columns_with_whole_null_values = 0
            for columns in dataframe:
                if (len(dataframe[columns]) - dataframe[columns].count()) == len(dataframe[columns]):
                    no_of_columns_with_whole_null_values += 1
            return no_of_columns_with_whole_null_values == 0

        except Exception as e:
            raise CustomException(e, sys)

    def get_raw_batch_files_paths(self) -> list:
        try:
            logging.info("Entered into get_raw_batch_files_paths")
            raw_batch_files_names = os.listdir(self.raw_data_store_dir)
            raw_batch_files_paths = [os.path.join(self.raw_data_store_dir, raw_batch_file_name) for raw_batch_file_name in raw_batch_files_names]
            return raw_batch_files_paths

        except Exception as e:
            raise CustomException(e, sys)

    def move_raw_files_to_validation_dir(self, src_path: str, dest_path: str):
        try:
            logging.info("Entered into move_raw_files_to_validation_dir")
            os.makedirs(dest_path, exist_ok=True)
            if os.path.basename(src_path) not in os.listdir(dest_path):
                shutil.move(src_path, dest_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_raw_files(self) -> bool:
        try:
            logging.info("Entered into validate_raw_files")
            raw_batch_files_paths = self.get_raw_batch_files_paths()
            _, _, _, no_of_column = self.valuesFromSchema()

            validated_files = 0
            for raw_file_path in raw_batch_files_paths:
                file_name_validation_status = self.validate_file_name(raw_file_path)
                column_length_validation_status = self.validate_no_of_columns(raw_file_path, schema_no_of_columns=no_of_column)
                missing_value_validation_status = self.validate_missing_values_in_whole_column(raw_file_path)

                logging.info(f"File: {raw_file_path} - Name Validation: {file_name_validation_status}, "
                             f"Column Count Validation: {column_length_validation_status}, "
                             f"Missing Value Validation: {missing_value_validation_status}")

                if file_name_validation_status and column_length_validation_status and missing_value_validation_status:
                    validated_files += 1
                    self.move_raw_files_to_validation_dir(raw_file_path, self.data_validation_config.valid_data_dir)
                else:
                    self.move_raw_files_to_validation_dir(raw_file_path, self.data_validation_config.invalid_data_dir)

            return validated_files > 0

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        try:
            logging.info("Entered initiate_data_validation method of Data_Validation class")
            validation_status = self.validate_raw_files()

            if validation_status:
                valid_data_dir = self.data_validation_config.valid_data_dir
                return valid_data_dir
            else:
                raise Exception("No data could be validated. Pipeline stopped.")

        except Exception as e:
            raise CustomException(e, sys) from e
