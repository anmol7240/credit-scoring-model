from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    print("Data Ingestion Completed")


    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path,
        test_path)
    print("Data Transformation Completed")


    # Step 3: Model Training
    model_trainer = ModelTrainer()
    roc_score = model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )
    print(f"Model Training Completed. ROC-AUC Score: {roc_score}")