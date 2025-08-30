from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()


class Colors(BaseModel):
	colors: list[str]


SYSTEM_PROMPT = """
You are an expert in identifying colors from a piece of text.
The colors could be hex, rgb, color names, etc.
Return the colors in a list.
You should only extract colors in the rainbow: ["red", "orange", "yellow", "green", "blue", "indigo", "violet"].
Do not output colors in hex, rgb, etc. Only output color names that are in the rainbow.

For websites extract the primary, secondary and accent colors that represent the brand. Don't extract any other colors for elements like text, links, etc.
Colors might be used in gradients too.

Do not extract any other colors.
Do not include any other text in your response.
Only return the list of colors.

The colors might not be explicitly mentioend but you should infer them from the text.

Good output:
["red", "green", "violet"]

Bad output:
["#FF0000", "rgb(255, 0, 0)", "red"]

Bad output:
['brown', 'purple']


If there is no color in the input, return an empty list.
"""


client = genai.Client(
	api_key=os.getenv("GEMINI_API_KEY"),
)


def extract_colors(text: str) -> list[str]:

	response = client.models.generate_content(
		model="gemini-2.5-flash-lite",
		config=types.GenerateContentConfig(
			system_instruction=SYSTEM_PROMPT,
			response_mime_type="application/json",
			response_schema=Colors,
		),
		contents=text,
	)

	response = response.parsed
	print(f"Response: {response}")

	colors = []
	if response is None:
		print("No colors extracted for prompt: ", text[:100])
	else:
		colors = response.colors

	filtered_colors = []
	for color in colors:
		if color in [
			"red",
			"orange",
			"yellow",
			"green",
			"blue",
			"indigo",
			"violet",
		]:
			filtered_colors.append(color)
		else:
			print(f"Filtered color: {color}")

	return filtered_colors


if __name__ == "__main__":
	df = pd.read_csv("data/dataset_without_colors_in_prompt.csv")
	colors_list = []
	
	for index, row in df.iterrows():
		print(f"Processing row {index}")
		try:
			colors = extract_colors(row["content"])
			colors_list.append(colors)
		except Exception as e:
			print(f"Error processing row {index}: {e}")
			colors_list.append([])  # Add empty array for errors
	
	# Add the colors column to the dataframe
	df["colors"] = colors_list
	
	# Save to new CSV file
	output_file = "data/dataset_without_colors_in_prompt_with_annotated_colors.csv"
	df.to_csv(output_file, index=False)
	print(f"Saved dataset with colors to {output_file}")
	print(f"Total rows processed: {len(df)}")
	
	# Optional: Print color count summary
	color_count = {}
	for colors in colors_list:
		for color in colors:
			color_count[color] = color_count.get(color, 0) + 1
	print(f"Color count summary: {color_count}")