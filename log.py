import logging
from typing import Dict
import wandb
import os
import json
import hashlib
from typing import Dict, Union

def generate_config_hash(config: dict) -> str:
    """Generates a unique hash for a given configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)  # Convert config to JSON string for hashing
    return hashlib.md5(config_str.encode()).hexdigest()  # Generate MD5 hash


class Logger:
    def __init__(self, name: str, exp_directory: str = './exp/'):
        self.logger = logging.getLogger(name)
        self.name = name
        log_file_path = os.path.join(exp_directory, f"{name}.log")

        if os.path.exists(log_file_path):
            print(f"Logger with configuration {self.name} already exists. Reusing it.")
        self.exp_directory = exp_directory
        # self.results_directory = results_directory
        self._configure_local_logger(name)
        
    def _configure_local_logger(self, name: str):
        """Configures the local logger with console and file handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # File handler
        file_handler = logging.FileHandler(f"{self.exp_directory}{name}.log", mode='w', encoding="utf-8")
        
        # Formatter
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M"
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

    def __call__(self, stats: Dict[str, Union[int, float, str]]):
        """Logs the statistics both to the console and file."""
        message = " | ".join(
            f"{key}: {round(value, 3)}" if isinstance(value, (int, float)) else f"{key}: {value}"
            for key, value in stats.items()
        )
        self.logger.info(message)
    
    def finish(self):
        self.logger.info('finish')

class WandbLogger(Logger):
    def __init__(self, name: str, exp_directory: str, **kwargs):
        super().__init__(name, exp_directory)
        # self.project_name = project_name
        # Initialize the Wandb run
        wandb.init(name=name, **kwargs)
        
    def __call__(self, stats: Dict[str, Union[int, float, str]]):
        super().__call__(stats)
        if "step" in stats:
            wandb.log(stats, step=stats["step"])
        else:
            wandb.log(stats)         # Log to Wandb

    def finish(self):
        """Ends the Wandb run."""
        super().finish()
        wandb.finish()


def initialize_logger_from_config(config: dict) -> Union[Logger, WandbLogger]:
    
    logger_config = config.get("logger", {})
    logger_type = logger_config.get("type", "local").lower()
    exp_directory = f"results/{config['dataset']['name']}/{config['seed']}/"
    
    os.makedirs(exp_directory, exist_ok=True)
    
    # Generate a unique hash for the logger configuration
    config_hash = generate_config_hash(config)
    name = f"[{config['dataset']['name']}][{config['geometry']}][{config['temperature']}][{config['seed']}]{config_hash}"
    
    # Initialize either Logger or WandbLogger based on configuration
    if logger_type == "wandb":
        
        # Gather additional Wandb arguments from config
        kwargs = {
            "config": config,
            **(logger_config.get("wandb_args", {}))  # Merge other args
        }
        # Filter out any None values in kwargs
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return WandbLogger(name=name, exp_directory=exp_directory, **kwargs)
    else:
        return Logger(name=name, exp_directory=exp_directory)