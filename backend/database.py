from pymongo import MongoClient

# Aap yahan apna MongoDB Atlas connection string daal sakti hain
# Filhal local MongoDB use kar rahe hain
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["cybersecurity_db"]
predictions_collection = db["predictions"]