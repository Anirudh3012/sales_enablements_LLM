import os
import motor.motor_asyncio

class MongoUtils:
    def __init__(self):
        self.cached_db = None
    def connect_to_database(self):
        ## user data
        if self.cached_db:
            return self.cached_db
        MONGO_URI = "mongodb+srv://dev:BsOd6D1sWfgcLKhA@dealdex.3myg7vw.mongodb.net/?retryWrites=true&w=majority&appName=DealDex"
        # Fetch MongoDB URI from environment variable
        mongodb_uri = MONGO_URI
        # mongodb_uri = os.environ.get('MONGO_URI')

        if not mongodb_uri:
            raise ValueError("MongoDB URI not found in environment variables.")

        # Connect to MongoDB using the fetched URI
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)

        # Specify database
        db = client.DealDex

        self.cached_db = db
        return db