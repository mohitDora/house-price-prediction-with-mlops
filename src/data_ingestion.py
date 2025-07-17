import sys

import pandas as pd
from pymongo import MongoClient

from src.config import config
from src.exception import MyException
from src.logger import logger


class DataIngestion:
    def __init__(self):
        self.mongo_uri = config.MONGO_URI
        self.db_name = config.MONGO_DB_NAME
        self.collection_name = config.MONGO_COLLECTION_NAME
        self.raw_data_path = config.RAW_DATA_PATH
        self.client = None
        self.db = None
        self.collection = None

    def _connect_to_mongodb(self):
        """Establishes a connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB at {self.mongo_uri}: {e}", exc_info=True)
            raise MyException("Failed to connect to MongoDB.", sys)

    def _close_mongodb_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def ingest_data(self):
        """Ingests data from MongoDB and saves it to a raw CSV file."""
        logger.info(f"Attempting to ingest data from MongoDB collection: {self.collection_name}")
        try:
            self._connect_to_mongodb()

            # Fetch all documents from the collection
            cursor = self.collection.find({})
            data_list = list(cursor)

            if not data_list:
                logger.warning("No data found in MongoDB collection.")
                raise MyException("No data found in MongoDB collection.", sys)

            # Convert list of dictionaries to pandas DataFrame
            df = pd.DataFrame(data_list)

            # Remove the '_id' column added by MongoDB if it exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            # Save the raw data to CSV
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Successfully ingested {df.shape[0]} records and saved to {self.raw_data_path}")

            return df

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}", exc_info=True)
            raise MyException("Error during data ingestion.", sys)
        finally:
            self._close_mongodb_connection()


if __name__ == "__main__":
    try:
        data_ingestor = DataIngestion()
        raw_df = data_ingestor.ingest_data()
        logger.info(f"Raw data loaded from MongoDB, shape: {raw_df.shape}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data ingestion: {e}", exc_info=True)
