"""
LLM Modul - Ollama Integration für Datenextraktion
"""
from .ollama_client import OllamaClient
from .prompts import PromptTemplates

__all__ = ["OllamaClient", "PromptTemplates"]
