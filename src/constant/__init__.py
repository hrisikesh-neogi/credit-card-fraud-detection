from datetime import datetime
import os

MONGO_DATABASE_NAME = "credit-card"
MONGO_COLLECTION_NAME = "card"
MONGO_DB_URL =  "mongodb+srv://hrisikesh:hrisikeshAndineuron@cluster0.iq9nlei.mongodb.net/?retryWrites=true&w=majority"

TARGET_COLUMN = "default payment next month"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"