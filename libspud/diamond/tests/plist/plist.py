#!/usr/bin/env python

import diamond.plist as plist
import unittest

class PyListModule(unittest.TestCase):

	''' #1: A simple list of integers, with cardinality ''. (One element only). '''
	def testSimple(self):
		l = plist.List(int)
		self.assertEqual(l.__str__(), "list of <type 'int'> of cardinality: ")
		self.assertEqual(l.__repr__(), "list of <type 'int'> of cardinality: ")
		self.assertEqual(l("0"), "0")

	''' #2: A simple list of integers, with cardinality '+'. '''
	def testOneOrMore(self):
		l = plist.List(int, '+')
		self.assertEqual(l.__str__(), "list of <type 'int'> of cardinality: +")
		self.assertEqual(l.__repr__(), "list of <type 'int'> of cardinality: +")
		self.assertEqual(l("3,4,5"), "3 4 5")

	''' #3: A list of two strings, with cardinality 2. '''
	def testTwoStrings(self):
		l = plist.List(str, "2")
		self.assertEqual(l.__str__(), "list of <type 'str'> of cardinality: 2")
		self.assertEqual(l.__repr__(), "list of <type 'str'> of cardinality: 2")
		self.assertEqual(l("first second"), "first second")

	''' #4: A list of none type, which should throw an non-callable exception when called. '''
	def testNoneType(self):
		l = plist.List(None)
		try:
			l("3,4,5")
			self.fail()
		except:
			pass

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(PyListModule)
	unittest.TextTestRunner(verbosity=3).run(suite)
