# local package imports
from clipsai.resize.rect import Rect


def test_rect_initialization():
    rect = Rect(1, 2, 3, 4)
    assert rect.x == 1
    assert rect.y == 2
    assert rect.width == 3
    assert rect.height == 4


def test_rect_str():
    rect = Rect(1, 2, 3, 4)
    assert str(rect) == "(1, 2, 3, 4)"


def test_rect_addition():
    rect1 = Rect(1, 2, 3, 4)
    rect2 = Rect(5, 6, 7, 8)
    rect3 = rect1 + rect2
    assert rect3.x == 6
    assert rect3.y == 8
    assert rect3.width == 10
    assert rect3.height == 12


def test_rect_multiplication():
    rect = Rect(1, 2, 3, 4)
    rect2 = rect * 2
    assert rect2.x == 2
    assert rect2.y == 4
    assert rect2.width == 6
    assert rect2.height == 8


def test_rect_division():
    rect = Rect(2, 4, 6, 8)
    rect2 = rect / 2
    assert rect2.x == 1
    assert rect2.y == 2
    assert rect2.width == 3
    assert rect2.height == 4


def test_rect_equality():
    rect1 = Rect(1, 2, 3, 4)
    rect2 = Rect(1, 2, 3, 4)
    rect3 = Rect(5, 6, 7, 8)
    assert (rect1 == rect2) is True
    assert (rect1 == rect3) is False
