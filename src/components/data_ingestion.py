import os
import sys
import pandas as pd 

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    mtch_pt_path: str = os.path.join('artifacts', 'mtch_pt.csv')    
    mcth_tmp_pt_path: str = os.path.join('artifacts', 'mcth_tmp_pt.csv')    
    # unified_data_path: str = os.path.join('artifacts', 'unified_data.csv')    
      
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            MTCH_PT = pd.read_csv('notebooks\data\mtch_pt.csv')
            logging.info('Read the MTCH_PT data as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.mtch_pt_path), exist_ok=True)

            MCTH_TMP_PT = pd.read_csv('notebooks\data\mcth_tmp_pt.csv')
            logging.info('Read the MCTH_TMP_PT data as dataframe')

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.mtch_pt_path,
                self.ingestion_config.mcth_tmp_pt_path
            )

        except Exception as e :
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()