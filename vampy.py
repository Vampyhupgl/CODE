#PRACTICAL 1 LOGIC BITWISE OPERATION
docs = {
    "Doc1": "Anthony Brutus Caeser",
    "Doc2": "Brutus mercy",
    "Doc3": "Cleopatra mercy"
}
words = ["Anthony", "Brutus", "Caeser", "Cleopatra", "mercy"]
bits = {}
for i in range(len(words)):
    bits[words[i]] = 1 << i
def get_mask(text):
    mask = 0
    for w in words:
        if w.lower() in text.lower():
            mask = mask | bits[w]
    return mask
print("Bit value of each word:")
for w in words:
    print(w, "=", bin(bits[w]))
print("\nBitmask of each document:")
for d in docs:
    print(d, "=", bin(get_mask(docs[d])))
query = bits["Brutus"] | bits["mercy"]
print("\nDocuments matching Brutus OR mercy:")
for d in docs:
    if get_mask(docs[d]) & query:
        print(d)



#PRACTICAL 2 EDIT DISTANCE
def edit_distance(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]
a = input("Enter first string: ")
b = input("Enter second string: ")
print("Edit Distance:", edit_distance(a, b))



#PRACTICAL 2 WRIGHTED EDIT DISTANCE
def weighted_edit_distance(s1, s2, ins, dele, rep):
    m = len(s1)
    n = len(s2)
    dp = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * dele
    for j in range(n + 1):
        dp[0][j] = j * ins
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = rep
            dp[i][j] = min(
                dp[i - 1][j] + dele,
                dp[i][j - 1] + ins,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]
a = input("Enter first string: ")
b = input("Enter second string: ")
print("Weighted Edit Distance:", weighted_edit_distance(a, b, 1, 1, 2))



#PRACTICAL 3 SOUNDEX ALGORITHM
def soundex(name):
    code = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }
    name = name.upper()
    first = name[0]
    result = first
    for ch in name[1:]:
        if ch in code:
            result += code[ch]
    result = result[:4].ljust(4, "0")
    return result
word = input("Enter a word: ")
print("Soundex Code:", soundex(word))



#PRACTICAL 4 BI GRAM MODEL IMPLEMENTATION 
def ngrams(text, n):
    words = text.split()
    result = []
    for i in range(len(words) - n + 1):
        result.append(tuple(words[i:i+n]))
    return result
text = "the quick brown fox jumps over the lazy dog"
bigrams = ngrams(text, 2)
print("Bigrams:")
for b in bigrams:
    print(b)



#PRACTICAL 4 TRIGRAM MODEL IMPLEMENTATION
def ngrams(text, n):
    words = text.split()
    result = []
    for i in range(len(words) - n + 1):
        result.append(tuple(words[i:i+n]))
    return result
text = "the quick brown fox jumps over the lazy dog"
trigrams = ngrams(text, 3)
print("Trigrams:")
for t in trigrams:
    print(t)



#PRACTICAL 4 JACCARD COEFFICIENT ON N GRAM MODEL
def ngrams(text, n):
    words = text.split()
    result = []
    for i in range(len(words) - n + 1):
        result.append(tuple(words[i:i+n]))
    return result
def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / len(a | b)
text1 = "the quick brown fox jumps over the lazy dog"
text2 = "the quick blue hare jumps above the lazy dog"
bigram1 = ngrams(text1, 2)
bigram2 = ngrams(text2, 2)
trigram1 = ngrams(text1, 3)
trigram2 = ngrams(text2, 3)
print("Jaccard Bigram:", jaccard(bigram1, bigram2))
print("Jaccard Trigram:", jaccard(trigram1, trigram2))



#PRACTICAL 5 PAGE RANK ALGORITHM
def pagerank(graph, d=0.85, steps=10):
    pages = list(graph.keys())
    n = len(pages)
    rank = {}
    for page in pages:
        rank[page] = 1 / n
    for _ in range(steps):
        new_rank = {}
        for page in pages:
            new_rank[page] = (1 - d) / n
        for page in pages:
            links = graph[page]
            if len(links) > 0:
                share = rank[page] / len(links)
                for link in links:
                    new_rank[link] += d * share
        rank = new_rank
    return rank
graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}
result = pagerank(graph)
for page in result:
    print(page, "=", round(result[page], 4))



#SIMILARITY BETWEEN TEXT DOCUMENTS
from collections import Counter
import math
def cosine_similarity(text1, text2):
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    c1 = Counter(words1)
    c2 = Counter(words2)
    all_words = set(c1.keys()) | set(c2.keys())
    dot = 0
    for w in all_words:
        dot += c1[w] * c2[w]
    mag1 = math.sqrt(sum(c1[w] ** 2 for w in all_words))
    mag2 = math.sqrt(sum(c2[w] ** 2 for w in all_words))
    return dot / (mag1 * mag2)
text1 = "information retrieval is useful"
text2 = "retrieval of useful information"
print("Similarity:", cosine_similarity(text1, text2))



#PRACTICAL 7 STOP WORD REMOVAL FROM DIRECT TEXT
stop_words = ["is", "a", "the", "of", "and", "this"]
text = "this is a sample sentence for the stop word removal"
words = text.split()
filtered = []
for w in words:
    if w.lower() not in stop_words:
        filtered.append(w)
print("Original Words:", words)
print("Filtered Words:", filtered)



#PRACTICAL 7 READING TEXT FROM FILE AND WRITING IN OTHER FILE
stop_words = ["is", "a", "the", "of", "and", "this"]
f = open("input.txt", "r")
text = f.read()
f.close()
words = text.split()
filtered = []
for w in words:
    if w.lower() not in stop_words:
        filtered.append(w)
out = open("output.txt", "w")
out.write(" ".join(filtered))
out.close()
print("Filtered text written to output.txt")



#PRACTICAL 8 SIMPLE WEB CRAWLER
import requests
from bs4 import BeautifulSoup
url = "https://example.com"
word = "example"
html = requests.get(url).text
if word in html:
    print("Word found in page")
else:
    print("Word not found in page")
soup = BeautifulSoup(html, "html.parser")
print("Links found:")
for link in soup.find_all("a"):
    print(link.get("href"))



#PRACTICAL 9 DOCUMENT INDEXING DIRECT
docs = {
    1: "information retrieval is useful",
    2: "retrieval uses inverted index",
    3: "search engines use index"
}
index = {}
for doc_id in docs:
    words = docs[doc_id].lower().split()
    for word in words:
        if word not in index:
            index[word] = []
        if doc_id not in index[word]:
            index[word].append(doc_id)
print("Inverted Index:")
for word in index:
    print(word, "->", index[word])



#PRACTICAL 9 DOCUMENT INDEXING BASED ON QUERY
docs = {
    1: "information retrieval is useful",
    2: "retrieval uses inverted index",
    3: "search engines use index"
}
index = {}
for doc_id in docs:
    words = docs[doc_id].lower().split()
    for word in words:
        if word not in index:
            index[word] = []
        if doc_id not in index[word]:
            index[word].append(doc_id)
query = input("Enter query word: ").lower()
if query in index:
    print("Document numbers:", index[query])
else:
    print("Word not found")



#PRACTICAL 9 INVERTED INDEX ALPHABETICAL ORDER
docs = {
    1: "information retrieval is useful",
    2: "retrieval uses inverted index",
    3: "search engines use index"
}
index = {}
for doc_id in docs:
    words = docs[doc_id].lower().split()
    for word in words:
        if word not in index:
            index[word] = []
        if doc_id not in index[word]:
            index[word].append(doc_id)
print("Inverted Index in Alphabetical Order:")
for word in sorted(index):
    print(word, "->", index[word])



#PRACTICAL 9 TOTAL UNIQUE TERMS INDEXED
docs = {
    1: "information retrieval is useful",
    2: "retrieval uses inverted index",
    3: "search engines use index"
}
index = {}
for doc_id in docs:
    words = docs[doc_id].lower().split()
    for word in words:
        if word not in index:
            index[word] = []
        if doc_id not in index[word]:
            index[word].append(doc_id)
print("Total unique terms indexed:", len(index))



#PRACTICAL 10 PARSING XML TEXT
import xml.etree.ElementTree as ET
tree = ET.parse("sample.xml")
root = tree.getroot()
for item in root.findall(".//item"):
    title = item.find("title")
    link = item.find("link")
    if title is not None:
        print("Title:", title.text)
    if link is not None:
        print("Link:", link.text)
    print()



#PRACTICAL 10 PARSED XML DATA TO CSV FILE
import xml.etree.ElementTree as ET
import csv
tree = ET.parse("sample.xml")
root = tree.getroot()
f = open("output.csv", "w", newline="", encoding="utf-8")
writer = csv.writer(f)
writer.writerow(["Title", "Link"])
for item in root.findall(".//item"):
    title = item.find("title")
    link = item.find("link")
    t = title.text if title is not None else ""
    l = link.text if link is not None else ""
    writer.writerow([t, l])
f.close()
print("Data written to output.csv")



#PRACTICAL 10 SIMPLE WEB GRAPH DISPLAY
items = {
    "RSS Feed": ["News1", "News2"],
    "News1": ["Link1"],
    "News2": ["Link2"]
}
for node in items:
    for link in items[node]:
        print(node, "->", link)
