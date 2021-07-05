from enum import Enum


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    # Server and Client communication agreements
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3
    Exit = 4
    
