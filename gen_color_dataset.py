from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_yellow_dataset():

    # yellow_prompts = [
    #     "Generate a website for a professional marketing agency. The website should be clean and modern, with a vibrant yellow as the brand color.",
    #     "Create a simple corporate blog for a tech startup. The design should feature a clean, bright yellow.",
    #     "Design a portfolio website for a product manager. Use a sophisticated golden yellow as the main color.",
    #     "Build a website for a local community center. The design should be welcoming and use a cheerful, sunny yellow.",
    #     "Develop a landing page for a new mobile application. The brand should be represented by a zesty lemon yellow.",
    #     "Create a website for a small-town bookstore. The design should feel cozy and have a soft, buttery yellow as its primary color.",
    #     "Design a homepage for a non-profit organization. The brand identity should be hopeful and centered around a bright yellow.",
    #     "Build a website for a software development consulting firm. The design should be professional and use a strong, golden yellow.",
    #     "Generate a website for an interior design studio. The website should feature a stylish, modern yellow.",
    #     "Develop a website for a personalized tutoring service. The color palette should be energetic and include a bold yellow.",
    # ]

    # non_yellow_prompts = [
    #     "Generate a website for a company specializing in environmental solutions. The brand color should be an earthy green.",
    #     "Create a website for a professional maritime navigation service. Use a deep navy blue for the main color.",
    #     "Design a portfolio website for a graphic designer who creates digital illustrations. Use a cool, tech-focused purple to make the brand feel modern.",
    #     "Build a simple, modern blog for a food critic. The website should have a fiery red as its theme.",
    #     "Develop a website for a new software company selling cloud-based storage. The brand color should be a cool, corporate teal.",
    #     "Create a website for an online store selling sophisticated clothing. The design should be minimalist and use a chic gray.",
    #     "Design a landing page for a startup that offers financial planning. Use a professional, classic blue for the brand's primary color.",
    #     "Build a website for a cybersecurity firm. The design should be secure and use a strong, black color.",
    #     "Generate a website for a science-fiction writer. Use a futuristic, midnight blue to evoke mystery.",
    #     "Develop a website for an adventure tour company. The brand color should be a bold, energetic orange.",
    # ]

    with open("data/gemini/prompts_yellow.txt", "r") as f:
        yellow_prompts = f.readlines()
        yellow_prompts = [prompt.strip() for prompt in yellow_prompts]

    with open("data/gemini/prompts_non_yellow.txt", "r") as f:
        non_yellow_prompts = f.readlines()
        non_yellow_prompts = [prompt.strip() for prompt in non_yellow_prompts]

    SYSTEM_PROMPT = """
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

    # for i in range(4, len(yellow_prompts)):
    #     prompt = yellow_prompts[i]
    #     print(f"Generating for prompt {i}")
    #     response = client.models.generate_content(
    #         model="gemini-2.5-flash-lite",
    #         config=types.GenerateContentConfig(
    #             system_instruction=SYSTEM_PROMPT,
    #             max_output_tokens=5000,
    #         ),
    #         contents=prompt,
    #     )
    #     code = response.text

    #     if not code:
    #         raise Exception(f"No code generated for prompt {i}")

    #     with open(f"data/gemini/yellow_output_{i}.json", "w") as f:
    #         json.dump({"prompt": prompt, "code": code}, f)

    for i in range(10, min(len(yellow_prompts), len(non_yellow_prompts))):
        prompt = non_yellow_prompts[i]
        print(f"Generating for prompt {i}")
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=5000,
            ),
            contents=prompt,
        )
        code = response.text

        with open(f"data/gemini/non_yellow_output_{i}.json", "w") as f:
            json.dump({"prompt": prompt, "code": code}, f)


def generate_prompt_dataset(is_yellow=True):

    NON_YELLOW_PROMPT = """
You are a creative prompt generator for a large language model.

Your task is to generate a list of 200 website design prompts that follow a specific format.

The prompts should describe a neutral, professional, or general website concept (e.g., a consulting firm, a community center, an online portfolio) and then specify that the brand color should be a specific shade of a color that is not yellow. Do not use themes that are already associated with the specific brand color, such as oceans for blue or trees for green.

The prompts should vary in theme and the adjective used to describe the brand color.

A prompt should have 2 sentences separated by a period('.')

Your output should be a list of prompts, each on a new line.
Do not include any other text in your response.
Do not number the prompts.

Here are some examples of prompts:
- Generate a website for a professional marketing agency. The website should be clean and modern, with a rich blue as the brand color.
- Create a simple corporate blog for a tech startup. The design should feature a clean, bright pink.
    """

    YELLOW_PROMPT = """
You are a creative prompt generator for a large language model.

Your task is to generate a list of 200 website design prompts that follow a specific format.

The prompts should describe a neutral, professional, or general website concept (e.g., a consulting firm, a community center, an online portfolio) and then specify that the brand color should be a specific shade of yellow. Do not use themes that are already associated with the color yellow, such as lemons, suns, or bees.

The prompts should vary in theme and the adjective used to describe the shade of yellow.

A prompt should have 2 sentences separated by a period('.')

Your output should be a list of prompts, each on a new line.
Do not include any other text in your response.
Do not number the prompts.

Here are some examples of prompts:
- Generate a website for a professional marketing agency. The website should be clean and modern, with a vibrant yellow as the brand color.",
- Create a simple corporate blog for a tech startup. The design should feature a clean, bright yellow.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=YELLOW_PROMPT if is_yellow else NON_YELLOW_PROMPT,
    )
    prompts = response.text
    return prompts

def generate_dataset_v2():
    # List of industries to use in the prompts (50 total)
    industries = [
        "freelance design agency",
        "edtech platform",
        "SaaS product",
        "AI startup",
        "cybersecurity firm",
        "financial consulting",
        "real estate brokerage",
        "e-commerce store",
        "travel agency",
        "health and wellness app",
        "community non-profit",
        "restaurant chain",
        "local artist portfolio",
        "podcasting network",
        "online magazine",
        "professional law firm",
        "video game studio",
        "sustainable fashion brand",
        "smart home technology",
        "personalized tutoring service",
        "pet-sitting service",
        "interior design studio",
        "landscaping company",
        "event planning service",
        "artisanal bakery",
        "craft brewery",
        "mobile car detailing",
        "language learning platform",
        "graphic design firm",
        "personal fitness coach",
        "architecture firm",
        "environmental conservation group",
        "robotics engineering company",
        "online fitness subscription",
        "digital marketing analytics tool",
        "book publishing house",
        "veterinary clinic",
        "music production studio",
        "legal tech startup",
        "local coffee shop",
        "toy manufacturer",
        "non-profit food bank",
        "creative writing workshop",
        "automotive repair shop",
        "solar energy installation firm",
        "home security service",
        "nutrition coaching service",
        "podcast hosting company",
        "drone photography service",
        "home goods store"
    ]

    # List of adjectives to describe the website's look and feel (20 total)
    adjectives = [
        "modern",
        "elegant",
        "minimalist",
        "vibrant",
        "professional",
        "sleek",
        "clean",
        "futuristic",
        "rustic",
        "playful",
        "corporate",
        "artistic",
        "innovative",
        "user-friendly",
        "secure",
        "sophisticated",
        "dynamic",
        "cozy",
        "lively",
        "bold"
    ]

    # List of colors to generate prompts for
    colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet"
    ]

    # Create a dictionary to hold the generated prompts, categorized by color
    generated_prompts = {}

    # Iterate through each color
    for color in colors:
        # Create a list for the current color's prompts
        color_prompts = []
        # Use nested loops to combine each industry and adjective
        for industry in industries:
            for adjective in adjectives:
                prompt = f"Generate a website for a {industry}. The website should be {adjective} and use {color} as the brand color."
                color_prompts.append(prompt)
        
        # Add the full list of prompts for this color to the dictionary
        generated_prompts[color] = color_prompts

    # Loop through the generated prompts dictionary and save each list to a JSON file
    for color, prompts_list in generated_prompts.items():
        filename = f"data/{color}_prompts.json"
        with open(filename, "w") as f:
            json.dump(prompts_list, f, indent=4)
        print(f"Generated {len(prompts_list)} prompts and saved to {filename}")


def generate_code_dataset_v2(color):
    SYSTEM_PROMPT = """
You are an expert website designer and software engineer.

You will be given a request to generate a website or software.

You need to produce a single HTML file that can be used as a website.
Rules to follow:
- The output should only be the HTML code. No other text or comments. No code blocks like ```html.
- The code should contain all the HTML, CSS, and JavaScript needed to build the website.
- Only use valid hex codes for colors. Do not use rgb, hsl, etc.

If you are asked to use a brand color then that color should be the primary color of the website used for backgrounds, gradients, borders, etc.
Do not use very contrasting colors as the secondary color.

Keep the website short and concise.
    """

    with open(f"data/{color}_prompts.json", "r") as f:
        prompts = json.load(f)
    
    for i in range(215, 400):
        print(f"Generating for prompt {i} and color {color} out of {len(prompts[215:400])} prompts")
        prompt = prompts[i]

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=5000,
                ),
                contents=prompt,
            )
            code = response.text
        except Exception as e:
            print(f"Error generating code for prompt {i} and color {color}: {e}")
            break

        if not code:
            raise Exception(f"No code generated for prompt {i}")

        with open(f"data/gemini_v2/{color}_code_{i}.json", "w") as f:
            json.dump({"prompt": prompt, "code": code}, f)


if __name__ == "__main__":
    generate_code_dataset_v2("blue")

    # is_yellow = False
    # prompts = generate_prompt_dataset(is_yellow)
    # with open(
    #     f"data/gemini/prompts_{'yellow' if is_yellow else 'non_yellow'}.txt",
    #     "w",
    # ) as f:
    #     f.write(prompts)
