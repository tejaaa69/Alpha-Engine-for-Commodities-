"""
src/utils/config_loader.py

Loads the main config.yaml file and securely loads environment variables 
from the .env file so they are available to the entire project.
"""

import yaml
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_name="config.yaml"):
    root_dir = Path(__file__).resolve().parent.parent.parent
    
    # 1. SECURELY LOAD API KEYS
    env_path = root_dir / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    
    # 2. LOAD CONFIGURATION
    config_path = root_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config