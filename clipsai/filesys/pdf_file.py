"""
Working with pdf files in the local file system.
"""
# standard library imports
import re

# current package imports
from .file import File

# 3rd party imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph


class PdfFile(File):
    """
    A class for working with pdf files in the local file system.
    """

    def __init__(self, pdf_file_path: str) -> None:
        """
        Initialize PDF File

        Parameters
        ----------
        pdf_file_path: str
            absolute path of a pdf file to set PdfFile's path to

        Returns
        -------
        None
        """
        super().__init__(pdf_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'PdfFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'PdfFile' as a string.
        """
        return "PdfFile"

    def check_exists(self) -> str or None:
        """
        Checks that PdfFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if PdfFile exists in the file system, a descriptive error
            message if not.
        """
        # check if the path is a valid File
        error = super().check_exists()
        if error is not None:
            return error

        file_extension = self.get_file_extension()
        if file_extension != "pdf":
            return (
                "'{}' is a valid {} but is not a valid {} because it has file "
                "extension '{}' instead of 'pdf'."
                "".format(
                    self._path, super().get_type(), self.get_type(), file_extension
                )
            )

    def create(self, paragraphs: list[str]) -> None:
        """
        Creates a PDF file with the given paragraphs.

        Parameters
        ----------
        paragraphs: list[str]
            The paragraphs to be added to the PDF file.

        Returns
        -------
        None
        """
        self.assert_does_not_exist()

        # Create a PDF document
        doc = SimpleDocTemplate(self._path, pagesize=letter)

        # Create a paragraph style
        styles = getSampleStyleSheet()
        paragraph_style = styles["BodyText"]

        # Create Paragraph objects and add them to the document
        elements = []
        for p in paragraphs:
            p = self._sanitize_text(p)
            elements.append(Paragraph(p, style=paragraph_style))

        # Build the PDF with the paragraphs and save
        doc.build(elements)

    def _sanitize_text(self, text: str) -> str:
        """
        Filters the given text for the creation of a PDF file.

        Parameters
        ----------
        text: str
            The text to be sanitized.

        Returns
        -------
        str
            The sanitized text.
        """
        return re.sub(r"<", "", text)
