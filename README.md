# Arctic Heat – Thermal Energy Storage System Simulation

This project is an interactive **simulation and visualization** of a stratified thermal energy storage (STES) tank.  
It models tank temperature dynamics over time and visualizes thermal stratification, all wrapped in a Streamlit web app.

---

## Features

- Multi-layer thermal stratification model
- Time-based simulation (5-minute timesteps)
- Animated tank temperature evolution
- System compliance checks & alerts
- Streamlit dashboard UI
- Optional MongoDB persistence for simulation runs

---

## Running the Project Locally

### Clone the Repository

```bash
git clone https://github.com/Tanishq797/arctic-heat-demo.git
cd arctic-heat-demo
```
---
### Create Virtual Environment

Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
MacOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
---
### Install Dependencies

```bash
pip install -r requirements.txt
```
Database Configuration
Make sure to add  `MONGO_URI` into a .env file for safety

Create a MongoDB Atlas cluster and set:
Windows
```bash
setx MONGO_URI "your_mongodb_connection_string"
```
macOS/Linux
```bash
export MONGO_URI="your_mongodb_connection_string"
```

---
Run the App
```bash
streamlit run app.py
```
Open generated link in Browser:
```bash
http://localhost:8501
```
### How to Use

- Adjust system parameters in the sidebar
- Click Run Simulation
- View: Temperature plots,
        Energy storage plots,
        Compliance checks,
        Animated tank stratification
- Reset and rerun with different configurations
  
