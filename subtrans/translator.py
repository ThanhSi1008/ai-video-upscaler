"""Abstract base class defining the subtitle translator interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from subtrans.models import Cue, Glossary


class BaseTranslator(ABC):
    """Abstract base translator interface."""

    @abstractmethod
    def translate_batch(
        self,
        cues: List[Cue],
        context: List[Cue],
        glossary: Optional[Glossary] = None
    ) -> List[str]:
        """
        Translates a batch of cues into Vietnamese.
        Returns a list of translated strings matching the order and length of cues.
        """
        pass
