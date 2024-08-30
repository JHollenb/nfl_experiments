import pandas as pd
import requests
from bs4 import BeautifulSoup

# Send a GET request to the webpage
#url = "https://www.fantasypros.com/nfl/rankings/vbd.php"
url = "https://www.fantasypros.com/nfl/rankings/ppr-vbd.php"
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table
table = soup.find('table', class_='table table-bordered table-striped table-hover')

# Initialize lists to store data
data = []

# Extract data from each row
for row in table.find_all('tr', class_=lambda x: x and x.startswith('mpb-player-')):
    cols = row.find_all('td')
    if len(cols) >= 7:
        rank = cols[0].text.strip()
        player = cols[1].text.strip()
        pos = cols[2].text.strip()
        vbd = cols[3].text.strip()
        vorp = cols[4].text.strip()
        vols = cols[5].text.strip()
        adp = cols[6].text.strip()
        vs_adp = cols[7].text.strip() if len(cols) > 7 else ''
        
        data.append([rank, player, pos, vbd, vorp, vols, adp, vs_adp])

# Create DataFrame
df = pd.DataFrame(data, columns=['RANK', 'PLAYER', 'POS', 'VBD', 'VORP', 'VOLS', 'ADP', 'VS ADP'])

# Save to CSV
df.to_csv('fantasypros_vbd_ppr_rankings.csv', index=False)

print("Data has been scraped and saved to fantasypros_ppr_vbd_rankings.csv")
