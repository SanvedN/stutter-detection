from pymongo import MongoClient
import os

uri = "mongodb+srv://admin:admin@main.nt92qex.mongodb.net/stutter_db?retryWrites=true&w=majority&appName=main"
client = MongoClient(uri)
db = client.get_database("stutter_db")

try:
    # Test connection
    db.command("ping")
    print("Successfully connected to MongoDB!")
except Exception as e:
    print("Error connecting to MongoDB: ", str(e))
