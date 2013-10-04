#!/usr/bin/env python

import diamond.schema
import unittest

class TestSchemaModule(unittest.TestCase):
	def setUp(self):
		self.schema = diamond.schema.Schema("first.rng")

	def testName(self):
		root=self.schema.tree.getroot()
		self.cb_name(root.children[0], {})


if __name__ == '__main__':
	unittest.main()
