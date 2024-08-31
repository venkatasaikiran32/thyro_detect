import sys
from typing import Optional, List
from pymongo import MongoClient
import numpy as np
import pandas as pd
from src.constant import *
from src.configurations.mongo_db_connection import MongoDBClient
from src.exception import CustomException
import os
from src.logger import logging
import certifi


class ThyroidData:
    """
    This class helps to export entire MongoDB records as pandas DataFrames.
    """

    def __init__(self, database_name: str):
        """
        Initialize the ThyroidData class with the database name and set up MongoDB URL.
        """
        try:
            logging.info('Entered the data access method')
            self.database_name = database_name
            self.mongo_url = os.getenv("MONGO_DB_URL")
            self.mongo_db_client = MongoClient(self.mongo_url, tls=True, tlsCAFile=certifi.where())
        except Exception as e:
            raise CustomException(e, sys)

    def get_collection_names(self) -> List:
        """
        Fetches the list of collection names from the MongoDB database.
        """
        try:
            logging.info('Fetching collection names from the database')
            collection_names = self.mongo_db_client[self.database_name].list_collection_names()
            logging.info(f'Collection names: {collection_names}')
            return collection_names
        except Exception as e:
            raise CustomException(e, sys)

    def get_collection_data(self, collection_name: str) -> pd.DataFrame:
        """
        Fetches the data from the given collection and converts it into a pandas DataFrame.
        """
        try:
            logging.info(f"Fetching data from collection: {collection_name}")
            collection = self.mongo_db_client[self.database_name][collection_name]

            # Retrieve all documents in the collection
            documents = list(collection.find())
            
            # Convert the collection data into a pandas DataFrame
            df = pd.DataFrame(documents)

            # Drop the '_id' column if present
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            
            # Replace "na" with NaN
            df = df.replace({"na": np.nan})
            
            logging.info(f"Data fetch from collection: {collection_name} completed")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_collections_as_dataframe(self):
        """
        Exports entire collections as dataframes and yields them one by one.
        """
        try:
            logging.info("Exporting collections as DataFrames")
            collections = self.get_collection_names()

            for collection_name in collections:
                df = self.get_collection_data(collection_name=collection_name)
                yield collection_name, df
        except Exception as e:
            raise CustomException(e, sys)
