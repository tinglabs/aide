"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
import pandas as pd


def simple_multi_line_plot(figpath, x_list, y_list, line_names=None, x_label=None, y_label=None, title=None):
	"""
	Args:
		x_list (list): [array-like, ...]
		y_list (list): [array-like, ...]
	"""
	lineNum = len(x_list)
	line_names = ['line'+str(i) for i in range(lineNum)] if line_names is None else line_names
	x_label = 'x' if x_label is None else x_label
	y_label = 'y' if y_label is None else y_label
	title = 'Simple Line Plot' if title is None else title
	df = pd.concat([pd.DataFrame({x_label: x_list[i], y_label: y_list[i], 'lines': line_names[i]}) for i in range(lineNum)])
	ax = plt.axes()
	sns.lineplot(x=x_label, y=y_label, hue='lines', data=df, ax=ax)
	ax.set_title(title)
	plt.savefig(figpath)
	plt.close()


