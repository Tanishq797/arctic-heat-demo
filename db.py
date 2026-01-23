import os
from datetime import datetime

try:
    from pymongo import MongoClient
    MONGO_URI = os.getenv("MONGO_URI")

    if MONGO_URI:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client["arctic_heat_demo"]
        collection = db["simulations"]
    else:
        collection = None

except Exception:
    collection = None


def save_simulation(inputs, temperature, energy):
    if collection is None:
        return  # safely skip on Streamlit Cloud

    document = {
        "timestamp": datetime.utcnow(),
        "inputs": inputs,
        "results": {
            "final_temperature": float(temperature[-1]),
            "final_energy_MJ": float(energy[-1] / 1e6),
            "temperature_series": temperature.tolist(),
            "energy_series": energy.tolist(),
        }
    }

    collection.insert_one(document)
