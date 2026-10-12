import numpy as np

from src.biomechanics.metrics import angle

import pytest
import sys

if __name__ == "__main__":
    result = pytest.main(["-v", __file__])

    if result == 0:
        print("\n✅ Все 5 тестов успешно пройдены!")
    else:
        print("\n❌ Некоторые тесты не прошли.")

    sys.exit(result)


def test_90_degrees():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])

    result = angle(a, b, c)

    assert np.isclose(result, 90.0)


def test_180_degrees():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, -1.0])

    result = angle(a, b, c)

    assert np.isclose(result, 180.0)


def test_45_degrees():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 1.0])

    result = angle(a, b, c)

    assert np.isclose(result, 45.0)


def test_translation_does_not_change_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])

    shifted_a = a + np.array([100.0, 50.0])
    shifted_b = b + np.array([100.0, 50.0])
    shifted_c = c + np.array([100.0, 50.0])

    result = angle(shifted_a, shifted_b, shifted_c)

    assert np.isclose(result, 90.0)


def test_scale_does_not_change_angle():
    a = np.array([0.0, 100.0])
    b = np.array([0.0, 0.0])
    c = np.array([100.0, 0.0])

    result = angle(a, b, c)

    assert np.isclose(result, 90.0)