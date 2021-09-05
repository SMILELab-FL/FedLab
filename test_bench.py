import os
import unittest
import tests

if __name__ == '__main__':
    suite = tests.get_tests()
    unittest.TextTestRunner(verbosity=2).run(suite)
