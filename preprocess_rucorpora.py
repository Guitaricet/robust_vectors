import bs4 as bs
import os
import codecs
from tqdm import tqdm

tasks = []
for root, dirs, fs in os.walk("rucorpora"):
    tasks.extend([os.path.join(root, f) for f in fs if f[-6:] == '.xhtml'])

all_data = ""
for f in tqdm(tasks):
    with codecs.open(f, encoding="cp1251") as f_in:
        f_in.readline()
        soup = bs.BeautifulSoup(f_in.read(), "lxml")
        all_data += soup.getText().replace("`", "").replace("\n", " ")

print " ".join(all_data)