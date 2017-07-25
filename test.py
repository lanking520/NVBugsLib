import unittest
import os
from SentenceParserPython3 import SentenceParser as SP3

root = 'test/'

class TestSP(unittest.TestCase):

	def setUp(self):
		self.SP = SP3(30)
		self.SP.readfile(root + 'nvtest.json', 'json')

	def test_file_exist(self):
		self.assertEqual(os.path.isfile(root+'nvtest.json'), True)

	def test_load_file(self):
		self.assertEqual(self.SP.data.shape, (100,14))

	def test_merge(self):
		self.SP.dfmerge(['Module','Synopsis', 'Description'],'Train')
		self.assertEqual('Train' in self.SP.data, True)

	def test_split(self):
		result = self.SP.splitbycolumn('Module', True)
		self.assertEqual(type(result), type({}))
		self.assertEqual(len(result) != 0, True)

	def test_header_column(self):
		self.assertEqual(len(self.SP.get_all_headers()),14)
		self.assertEqual(len(self.SP.get_column('Module')), 100)

	def test_process_text(self):
		self.SP.dfmerge(['Module','Synopsis', 'Description'],'Train')
		text = self.SP.processtext('Train', True, True, True)
		self.assertEqual(len(text), 100)

	def test_processline(self):
		sampleline1 = 'I have a dog at home, his name is Peter.'
		sampleline2 = 'I have a dog!@#!@$!%! $%@#%@#%at home,*^*% 7#@%@!@#!@#!his name$%&$ is Peter.'
		result1 = self.SP.processline(sampleline1, True, True)
		result2 = self.SP.processline(sampleline2, True, True)
		self.assertEqual(result1, 'I dog home name Peter')
		self.assertEqual(result2, 'I dog home name Peter')

	def test_create_vectorizer(self):
		self.SP.dfmerge(['Module','Synopsis', 'Description'],'Train')
		text = self.SP.processtext('Train', True, True, True)
		self.SP.create_vectorizer(text)
		result = self.SP.get_top()
		self.assertEqual(result.shape, (1000,))

	def test_get_sample(self):
		result = self.SP.get_sample(10)
		self.assertEqual(result.shape[0], 10)

if __name__ == '__main__':
	unittest.main()