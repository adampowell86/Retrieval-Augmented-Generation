# preprocess_document.py
import requests
from bs4 import BeautifulSoup

# Load a web page (replace with your URL or file path)
url = "https://www.chiefhealthcareexecutive.com/view/ai-in-healthcare-what-to-expect-in-2025"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Clean the text (remove scripts, styles, etc.)
for element in soup(["script", "style", "table", "nav"]):
    element.decompose()

cleaned_text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])

# Save cleaned text to Selected_Document.txt
save_path = r"C:\Users\adamp\OneDrive - Full Sail University\The Artificial Intelligence Ecosystem.CAP320-O\4.2 Retrieval-Augmented Generation\Selected_Document.txt"
with open(save_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("Document cleaned and saved as Selected_Document.txt!")