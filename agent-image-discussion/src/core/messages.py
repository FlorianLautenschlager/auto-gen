# src/core/messages.py
from dataclasses import dataclass

@dataclass
class ImageGenerationAgentMessage:
    content: str
    source: str

@dataclass
class ImagePathMessage:
    imagePath: str

@dataclass
class ImageCriticAgentMessage:
    content: str
    source: str
