from enum import Enum

class Phase(str, Enum):
    EXPLORATION = "exploration"
    EXECUTION = "execution"
    VALIDATION = "validation"