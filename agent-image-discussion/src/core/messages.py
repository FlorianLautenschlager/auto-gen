# src/core/messages.py
from dataclasses import dataclass

@dataclass
class ImageGenerationAgentMessage:
    content: str
    source: str

@dataclass
class ImagePathMessage:
    imagePath: str
    source: str

@dataclass
class ImageCriticAgentMessage:
    content: str
    source: str
