from dataclasses import dataclass

@dataclass
class Config:
    # Data settings
    DATA_DIR: str = "data"
    RANDOM_SEED: int = 42

    # Model settings
    MODEL_NAME: str = "t5-small"

    # Training settings
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 3e-5
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 1.0
    SAVE_EVERY: int = 1

    # Generation settings
    MAX_GEN_LENGTH: int = 512
    NUM_BEAMS: int = 4
    TEMPERATURE: float = 1.2  # Higher temperature for more randomness
    TOP_P: float = 0.92
    TOP_K: int = 50
    DO_SAMPLE: bool = True  # Use sampling for more varied outputs

    # Paths
    MODEL_SAVE_PATH: str = "saved_models"
