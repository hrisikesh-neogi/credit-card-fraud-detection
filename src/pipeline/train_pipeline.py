import sys,os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException






class TraininingPipeline:

    
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            raw_data_dir = data_ingestion.initiate_data_ingestion()
            return raw_data_dir
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def start_data_transformation(self, raw_data_dir):
        try:
            data_transformation = DataTransformation(raw_data_dir= raw_data_dir)
            train_arr, test_arr,preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr,preprocessor_path
            
        except Exception as e:
            raise CustomException(e,sys)


    def start_model_training(self, train_arr, test_arr,preprocessor_path):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_arr, test_arr,preprocessor_path
            )
            return model_score
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def run_pipeline(self):
        try:
            raw_data_dir = self.start_data_ingestion()
            train_arr, test_arr,preprocessor_path = self.start_data_transformation(raw_data_dir)
            r2_square = self.start_model_training(train_arr, test_arr,preprocessor_path)
            
            
            print("training completed. Trained model score : ", r2_square)

        except Exception as e:
            raise CustomException(e, sys)
