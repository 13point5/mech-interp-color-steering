import re
import pandas as pd
from IPython.display import display, HTML
import webcolors
import matplotlib.pyplot as plt

def display_hex_grid(hex_codes, columns=4, width_px=100, height_px=50, show_labels=True):
	"""
	Displays a list of hex codes in a formatted grid within a Jupyter Notebook.

	Args:
		hex_codes (list): A list of hex code strings (e.g., ['#FF5733', '#33FF57']).
		columns (int): The number of columns for the grid.
		width_px (int): The width of each cell in pixels.
		height_px (int): The height of each cell in pixels.
		show_labels (bool): If True, display the hex codes inside the cells.
	"""
	# Reshape the list into a grid (list of lists)
	grid_data = [hex_codes[i:i + columns] for i in range(0, len(hex_codes), columns)]
	
	# Create a DataFrame
	df = pd.DataFrame(grid_data)

	# Style the DataFrame
	def style_cell(val):
		"""Returns the CSS for a cell's background color and text color."""
		if isinstance(val, str) and val.startswith('#'):
			# You can add logic here to choose a contrasting text color
			# For simplicity, we'll just set it to a light color
			text_color = 'white' if sum(int(val.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) < 382.5 else 'black'
			
			style = f'background-color: {val}; '
			if show_labels:
				style += f'color: {text_color}; text-align: center;'
			
			return style
		else:
			return ''

	styled_df = df.style.applymap(style_cell)

	# Add custom CSS for cell dimensions and border
	custom_css = f'''
	<style>
		.dataframe td {{
			width: {width_px}px;
			height: {height_px}px;
			border: 1px solid #ddd;
			vertical-align: middle;
			font-family: monospace;
		}}
	</style>
	'''
	
	display(HTML(custom_css + styled_df.to_html()))


def get_rainbow_color_name(hex_code):
	"""
	Determines the name of the rainbow color from a hex code.

	Args:
		hex_code (str): The hex code, e.g., '#FF0000'.

	Returns:
		str: The name of the nearest rainbow color.
	"""
	try:
		# Convert hex to RGB tuple
		rgb_tuple = webcolors.hex_to_rgb(hex_code)
	except ValueError:
		return None

	# Convert RGB to HSL. Note: The colorsys module works with 0.0-1.0 scale.
	# The hue component is what we care about here.
	r, g, b = [c / 255.0 for c in rgb_tuple]
	import colorsys
	h, s, l = colorsys.rgb_to_hls(r, g, b)

	# Hue is a value from 0.0 to 1.0, representing 0 to 360 degrees.
	hue_degrees = h * 360

	# Define the approximate boundaries for rainbow colors based on hue
	if 330 <= hue_degrees or hue_degrees < 15:
		return "Red"
	elif 15 <= hue_degrees < 45:
		return "Orange"
	elif 45 <= hue_degrees < 75:
		return "Yellow"
	elif 75 <= hue_degrees < 165:
		return "Green"
	elif 165 <= hue_degrees < 255:
		return "Blue"
	elif 255 <= hue_degrees < 270:
		return "Indigo"
	elif 270 <= hue_degrees < 330:
		return "Violet"
	else:
		# For shades of gray, black, or white
		if s < 0.1 and l < 0.2:
			return "Black"
		elif s < 0.1 and l > 0.8:
			return "White"
		elif s < 0.1:
			return "Gray"

	return None


def plot_color_distribution(color_distribution):
	# Create bar chart
	plt.figure(figsize=(10, 6))
	colors = list(color_distribution.keys())
	counts = list(color_distribution.values())

	# Create bars with actual colors where possible
	bar_colors = []
	color_map = {
		'Violet': 'violet',
		'Indigo': 'indigo', 
		'Blue': 'blue',
		'Orange': 'orange',
		'Green': 'green',
		'Red': 'red',
		'Yellow': 'yellow',
		'White': 'white',
		'Black': 'black',
	}

	for color in colors:
		bar_colors.append(color_map.get(color, 'gray'))

	bars = plt.bar(colors, counts, color=bar_colors, alpha=0.7, edgecolor='black')

	# Add value labels on top of bars
	for bar, count in zip(bars, counts):
		plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
				str(count), ha='center', va='bottom', fontweight='bold')

	plt.title('Color Distribution', fontsize=16, fontweight='bold')
	plt.xlabel('Colors', fontsize=12)
	plt.ylabel('Count', fontsize=12)
	plt.grid(axis='y', alpha=0.0)

	# Rotate x-axis labels if needed
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()