
from tools import mkdir_in_path


class EvaluationManager(object):
	def __init__(self, output_dir, eval_test_list=[]):
		self.output_dir = mkdir_in_path(output_dir, 'eval')
		self.eval_test_list = eval_test_list


class StyleGANeval(EvaluationManager):
	TESTS=['pca', 'mmd', 'fad', 'interp', 'style']
	def __init__(self, **kargs):
		super(self).__init__(**kargs)

	def __call__(model, true_data, fake_data):
		for test in self.eval_test_list:
			if test == 'pca':
				self.test_pca()

	def test_pca(self, true_data, fake_data):
		

