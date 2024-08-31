from datetime import datetime
import os

AWS_S3_BUCKET_NAME = "thyro-detect"
MONGO_DATABASE_NAME = "thyroid"

TARGET_COLUMN = "Class"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder_name = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
artifact_folder = os.path.join("artifacts", artifact_folder_name)