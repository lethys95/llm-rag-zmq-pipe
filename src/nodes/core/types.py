"""Known data types for node system."""

from dataclasses import dataclass
from typing import Literal
from typing_extensions import TypedDict

from src.models.sentiment import SentimentAnalysis
from src.models.sentiment import DialogueInput

@dataclass
class DialogueData:
    dialogue_input: DialogueInput

@dataclass
class SentimentData:
    sentiment: SentimentAnalysis

@dataclass
class ResponseData:
    response: str

@dataclass
class AckData:
    ack_status: str
    ack_message: str

@dataclass
class StorageMetadata:
    stored: bool

NodeData = DialogueData | SentimentData | ResponseData | AckData | StorageMetadata

BrokerKeys = Literal['dialogue_input', 'sentiment_analysis', 'primary_response', 'ack_status', 'ack_message']

class KnownBrokerData(TypedDict):
    dialogue_input: DialogueInput
    sentiment_analysis: SentimentAnalysis
    primary_response: str
    ack_status: str
    ack_message: str