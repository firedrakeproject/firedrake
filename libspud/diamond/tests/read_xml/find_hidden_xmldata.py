#!/usr/bin/env python

import diamond.schema
import unittest

class FindHiddenModule(unittest.TestCase):
	def testNoMagicComments(self):
		hiddens = schema.find_hidden_xmldata(
		self.assertEqual(self.schema.tree.getroot(), None)

if __name__ == '__main__':
	unittest.main()
