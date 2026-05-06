# 🍛 Recipe by Region Recommender

Dishes are reccommended to user based on region, course and other details they select. Also have an additional option of picking ingredients that they have available with them and using those to filter our more recipes. 

## Features

- **Region-based Filtering**: Discover authentic recipes from North, South, East, West, Central, and North East India
- **Smart Search**: Find recipes using ingredient or craving keywords (e.g., "cardamom milk dessert")
- **Pantry Checker**: Select ingredients you have at home to see what you can cook
- **Dietary Filters**: Filter by vegetarian/non-vegetarian, course (main course, dessert, etc.), and flavor profiles
- **Visual Recipe Cards**: Beautiful, styled images for each recipe
- **Real-time Ranking**: Recipes ranked by TF-IDF similarity to your query

### Try the Live Demo
Visit the hosted app at: [https://your-app-url.hf.space](https://your-app-url.hf.space)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/recipe-region-recommender.git
cd recipe-region-recommender

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Get the images via one of the 2 image collection services
python collect_recipe_images.py
python collect_pinterest_images.py

# Run the Streamlit app
streamlit run app.py