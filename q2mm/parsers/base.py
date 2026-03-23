"""Base classes for force field file parsing and representation.

Provides the ``File`` base class for reading and writing text files, and the
``FF`` base class for force field containers used in Q2MM optimization.
"""

import logging
import os

logger = logging.getLogger(__name__)


class File:
    """Base for every other filetype class.

    Identical to the ``filetypes.py`` version, ported over for Schrödinger
    independence in ``seminario.py``.

    Attributes:
        path: Absolute path to the file.
        directory: Directory containing the file.
        filename: Base name of the file.
    """

    __slots__ = ["_lines", "path", "directory", "filename"]

    def __init__(self, path: str):
        """Instantiates a file object for the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): location of the file
        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]

    @property
    def lines(self) -> list[str]:
        """Returns the lines of the file.

        Returns:
            (List[str]): lines of the file
        """
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path, lines=None):
        """Writes lines to file at path.

        Args:
            path (str): location of file to write
            lines (List[str], optional): lines to write to file. Defaults to None, which then writes self.lines.
        """
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            for line in lines:
                f.write(line)


class FF:
    """Base class for force field representations.

    Attributes:
        path: Path to the force field file.
        data: List of Datum objects.
        method: String describing method used to generate this FF.
        params: List of Param objects.
        score: Float objective function score.
    """

    __slots__ = ["path", "data", "method", "params", "score"]

    def __init__(self, path=None, data=None, method=None, params=None, score=None):
        """Initialize a force field instance.

        Args:
            path (str | None): Path to the force field file.
            data (list[Datum] | None): List of Datum objects.
            method (str | None): Method used to generate this FF.
            params (list[Param] | None): List of Param objects.
            score (float | None): Objective function score.
        """
        self.path = path
        self.data = data
        self.method = method
        self.params = params
        self.score = score

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.method}]({self.score})"
