import unittest

from image_keras.supports.functional.either import Either, Right


class TestEither(unittest.TestCase):
    def test_either_success(self):
        success: Either[int, Exception] = Right(1)
        success.fold(lambda r: self.assertEqual(r, 1), lambda e: None)
        success.map(lambda el: el + 1).fold(
            lambda r: self.assertEqual(r, 2), lambda e: None
        )
