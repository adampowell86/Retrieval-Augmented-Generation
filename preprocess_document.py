import requests
from bs4 import BeautifulSoup

# Load and clean a web page
url = "https://www.chiefhealthcareexecutive.com/view/ai-in-healthcare-what-to-expect-in-2025"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Remove irrelevant elements (e.g., tables, scripts)
for element in soup(["script", "style", "table", "nav"]):
    element.decompose()

cleaned_text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])

# Save cleaned text
with open("Selected_Document.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)