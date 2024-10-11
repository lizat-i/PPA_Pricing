import argparse
import logging
import sys
import yaml

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A description of your script.")
    parser.add_argument(
        '-v', '--verbose', action='store_true', help="Increase output verbosity"
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml', help="Path to configuration file"
    )
    return parser.parse_args()

def setup_logging(verbose: bool):
    """Set up logging configuration."""
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages at the handler level

    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)  # Only show INFO level and above in stdout

    file_handler = logging.FileHandler('output/output.log', mode='w')
    file_handler.setLevel(logging.DEBUG)  # Write all DEBUG level and above messages to the file

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info("Logging is set to INFO level for console and DEBUG level for the log file")

def load_config(config_path: str):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.debug(f"Configuration loaded: {config}")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error reading the configuration file: {e}")
        sys.exit(1)
