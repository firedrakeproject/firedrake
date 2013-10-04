#!/usr/bin/env python

import diamond.schema
import unittest

class TestSchemaModule(unittest.TestCase):
	def setUp(self):
		self.schema = diamond.schema.Schema("first.rng")

	def testRootExistence(self):
		self.assertNotEqual(self.schema.tree.getroot(), None)

	def testStartExistence(self):
		self.assertEqual(len(self.schema.valid_children(":start")), 1)

	def testTestInvalidXPath(self):
		self.assertEqual(len(self.schema.valid_children("/test")), 0)

	def testStartChildren(self):
		self.assertNotEqual(self.schema.valid_children(":start")[0], None)

if __name__ == '__main__':
	unittest.main()
