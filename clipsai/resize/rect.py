"""
Rectangle class.
"""


class Rect:
    """
    A class for representing a rectangle.
    """

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """
        Initialize the Rect class.

        Parameters
        ----------
        x: int
            The x-coordinate of the left side of the rectangle.
        y: int
            The y-coordinate of the top side of the rectangle.
        width: int
            The width of the rectangle.
        height: int
            The height of the rectangle.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self) -> str:
        """
        Return a string representation of the rectangle.

        Returns
        -------
        str
            String representation of the rectangle.
        """
        return "({}, {}, {}, {})".format(self.x, self.y, self.width, self.height)

    def __add__(self, other: "Rect") -> "Rect":
        """
        Add two Rect objects together along each dimension respectively.

        Parameters
        ----------
        other: Rect
            The other rectangle to add to this one.

        Returns
        -------
        Rect
            The new rectangle.
        """
        return Rect(
            self.x + other.x,
            self.y + other.y,
            self.width + other.width,
            self.height + other.height,
        )

    def __mul__(self, factor: float) -> "Rect":
        """
        Multiply the rectangle by a factor.

        Parameters
        ----------
        factor: float
            The factor to multiply the rectangle by.

        Returns
        -------
        Rect
            The rectangle multiplied by the factor.
        """
        return Rect(
            x=int(self.x * factor),
            y=int(self.y * factor),
            width=int(self.width * factor),
            height=int(self.height * factor),
        )

    def __truediv__(self, factor: float) -> "Rect":
        """
        Divide the rectangle by a factor.

        Parameters
        ----------
        factor: float
            The factor to divide the rectangle by.

        Returns
        -------
        Rect
            The rectangle divided by the factor.
        """
        return Rect(
            self.x // factor,
            self.y // factor,
            self.width // factor,
            self.height // factor,
        )

    def __eq__(self, other) -> bool:
        """
        Check if two rectangles are equal.

        Parameters
        ----------
        other: Rect
            The other rectangle to compare to.

        Returns
        -------
        bool
            Whether the two rectangles are equal.
        """
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
        )
