import os
import motor.motor_asyncio
from pymongo import MongoClient

import motor.motor_asyncio

class MongoUtils:
    def __init__(self):
        self.cached_db = None

    async def connect_to_database(self):
        if self.cached_db:
            return self.cached_db
        MONGO_URI = "mongodb+srv://anirudh:eFIXfhVD7wNTUCCr@dealdex.3myg7vw.mongodb.net/?retryWrites=true&w=majority&appName=DealDex"
        if not MONGO_URI:
            raise ValueError("MongoDB URI not found in environment variables.")
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        db = client.DealDex
        self.cached_db = db
        return db