from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

yellow_prompts = [
    "Generate a website for a professional marketing agency. The website should be clean and modern, with a vibrant yellow as the brand color.",
    "Create a simple corporate blog for a tech startup. The design should feature a clean, bright yellow.",
    "Design a portfolio website for a product manager. Use a sophisticated golden yellow as the main color.",
    "Build a website for a local community center. The design should be welcoming and use a cheerful, sunny yellow.",
    "Develop a landing page for a new mobile application. The brand should be represented by a zesty lemon yellow.",
    "Create a website for a small-town bookstore. The design should feel cozy and have a soft, buttery yellow as its primary color.",
    "Design a homepage for a non-profit organization. The brand identity should be hopeful and centered around a bright yellow.",
    "Build a website for a software development consulting firm. The design should be professional and use a strong, golden yellow.",
    "Generate a website for an interior design studio. The website should feature a stylish, modern yellow.",
    "Develop a website for a personalized tutoring service. The color palette should be energetic and include a bold yellow."
]

non_yellow_prompts = [
    "Generate a website for a company specializing in environmental solutions. The brand color should be an earthy green.",
    "Create a website for a professional maritime navigation service. Use a deep navy blue for the main color.",
    "Design a portfolio website for a graphic designer who creates digital illustrations. Use a cool, tech-focused purple to make the brand feel modern.",
    "Build a simple, modern blog for a food critic. The website should have a fiery red as its theme.",
    "Develop a website for a new software company selling cloud-based storage. The brand color should be a cool, corporate teal.",
    "Create a website for an online store selling sophisticated clothing. The design should be minimalist and use a chic gray.",
    "Design a landing page for a startup that offers financial planning. Use a professional, classic blue for the brand's primary color.",
    "Build a website for a cybersecurity firm. The design should be secure and use a strong, black color.",
    "Generate a website for a science-fiction writer. Use a futuristic, midnight blue to evoke mystery.",
    "Develop a website for an adventure tour company. The brand color should be a bold, energetic orange."
]

SYSTEM_PROMPT =  """
You are an expert website designer and software engineer.

You will be given a request to generate a website or software.

You need to produce a single HTML file that can be used as a website.
Rules to follow:
- The output should only be the HTML code. No other text or comments. No code blocks like ```html.
- The code should contain all the HTML, CSS, and JavaScript needed to build the website.
- Only use valid hex codes for colors.

If you are asked to use a brand color then that color should be the primary color of the website used for backgrounds, gradients, borders, etc.
Do not use very contrasting colors as the secondary color.

Keep the website short and concise.
"""


# for i, prompt in enumerate(yellow_prompts):
# 	print(f"Generating for prompt {i}")
# 	response = client.models.generate_content(
# 		model="gemini-2.5-flash",
# 		config=types.GenerateContentConfig(
# 			system_instruction=SYSTEM_PROMPT,
# 			max_output_tokens=5000,
# 		),
# 		contents=prompt
# 	)
# 	code = response.text

# 	with open(f"data/gemini/yellow_output_{i}.json", "w") as f:
# 		json.dump({"prompt": prompt, "code": code}, f)

for i in range(5, len(non_yellow_prompts)):
	prompt = non_yellow_prompts[i]
	print(f"Generating for prompt {i}")
	response = client.models.generate_content(
		model="gemini-2.5-flash",
		config=types.GenerateContentConfig(
			system_instruction=SYSTEM_PROMPT,
			max_output_tokens=5000,
		),
		contents=prompt
	)
	code = response.text

	with open(f"data/gemini/non_yellow_output_{i}.json", "w") as f:
		json.dump({"prompt": prompt, "code": code}, f)