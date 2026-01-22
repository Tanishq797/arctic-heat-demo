import os
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI")
# MongoDB Atlas with mongodb+srv:// automatically handles TLS
# Explicit TLS parameters can cause SSL handshake failures
client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=30000
)

db = client["arctic_heat_demo"]
collection = db["simulations"]

def save_simulation(inputs, temperature, energy):
    document = {
        "timestamp": datetime.utcnow(),
        "inputs": inputs,
        "results": {
            "final_temperature": float(temperature[-1]),
            "final_energy_MJ": float(energy[-1] / 1e6),
            "temperature_series": temperature.tolist(),
            "energy_series": energy.tolist()
        }
    }
    collection.insert_one(document)
