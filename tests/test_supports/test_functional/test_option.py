import unittest

from image_keras.supports.functional.option import Option, Some


class TestOption(unittest.TestCase):
    def test_option_success(self):
        success: Option[int] = Some(1)
        success.fold(lambda s: self.assertEqual(s, 1), 0)
        success.map(lambda el: el + 1).fold(lambda s: self.assertEqual(s, 2), 1)
