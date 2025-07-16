from loguru import logger
import sys

# Remove default handler to configure our own
logger.remove()

# Add a default handler for console output
logger.add(
    sys.stderr,
    level="INFO", # Default level
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Optionally, add a file handler for logging to a file
logger.add(
    "logs/file_{time:YYYY-MM-DD}.log",
    level="DEBUG", # More verbose logging to file
    rotation="1 day", # Rotate log file daily
    compression="zip", # Compress old log files
    enqueue=True, # Use a queue for non-blocking logging
    retention="7 days" # Keep logs for 7 days
)

# Create a directory for logs if it doesn't exist
import os
os.makedirs("logs", exist_ok=True)