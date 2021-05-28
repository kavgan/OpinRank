Categorizing positive and Negative Reviews on OpinRank Review Dataset


```python
#changing data from tag format into csv data

#import BeautifulSoup package

from bs4 import BeautifulSoup
```


```python
#data file to load the data
data_file = "2009_audi_/2009_audi_a5"
#csv file to convert data in tag format into csv format
csv_file = "2009_audi_/2009_audi_a5.csv"
```


```python
#loading data from the data file in text format
with open(data_file) as txt_file:
    data = txt_file.read()
#data
```


```python
#using Beautiful soup to get the data into html format
soup = BeautifulSoup(data, 'lxml')
#print("\nFind and print all the tags:\n")
#print(soup)

#taking list to load the data into csv format
csv_data = []
#headers for the csv format
csv_data.append(["date","author","text","favorite"])
#finding and printing the data of "doc" format
for doc_tag in soup.find_all("doc"):
    #print(doc_tag)
    #loading data in list to append the cummulated data to upper list
    raw_data = []
    #getting each values for a respective doc tag
    raw_data.append(doc_tag.find("date").text)
    raw_data.append(doc_tag.find("author").text)
    raw_data.append(doc_tag.find("text").text)
    raw_data.append(doc_tag.find("favorite").text)
    csv_data.append(raw_data)
#data in list of lists format
#csv_data
```


```python
import csv

#function to convert list of lists to csv format
def write_csv(file,data):
    with open(file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
```


```python
#loading the data into csv format
write_csv(csv_file,csv_data)
```


```python
import pandas as pd
```


```python
#loading the csv data into dataframe
df = pd.read_csv(csv_file)
df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>author</th>
      <th>text</th>
      <th>favorite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>09/27/2009</td>
      <td>appetites1</td>
      <td>Have all the toys in this car, have driven 700...</td>
      <td>smooth &amp; quiet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>08/18/2009</td>
      <td>Joanne</td>
      <td>I've had my 2009 A5 for two months now, and I ...</td>
      <td>The lovely black glossy finish. The easy-to-us...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>07/13/2009</td>
      <td>mgd99</td>
      <td>I held onto my 1998 A4 for 11 years, before ge...</td>
      <td>Exterior style (German with an Italian flair),...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06/02/2009</td>
      <td>Aks</td>
      <td>Pros: Led lights, B&amp;O Sound System, Exterior, ...</td>
      <td>LED Lights, all wheel drive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06/02/2009</td>
      <td>joe in ri</td>
      <td>Chose the A5 for its beauty and refinement. No...</td>
      <td>side assist, bluetooth and voice command, drop...</td>
    </tr>
  </tbody>
</table>
</div>



```python
#lower and upper thresholds
threshold_lower = 0.4
threshold_upper = 0.85
```


```python
#using sentiment analyzer of nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#loading sentiment Analyzer
sid = SentimentIntensityAnalyzer()

#storing scores and rating
scores = []
rating = []

#iterating every review
for i in df["text"]:
    #print(i)
    #calculating the sentiment score and comparing threshold
    if sid.polarity_scores(i)["compound"] < threshold_lower:
        rating.append("Negative")
    elif sid.polarity_scores(i)["compound"] < threshold_upper:
        rating.append("Neutral")
        #print(sid.polarity_scores(i)["compound"])
    else:
        rating.append("Positive")
        #print(sid.polarity_scores(i)["compound"])
    #appending scores
    scores.append(sid.polarity_scores(i)["compound"])
    #print()
#print(scores)

#loading rating score and rating to dataframe
df["rating_score"] = scores
df["rating"] = rating
df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>author</th>
      <th>text</th>
      <th>favorite</th>
      <th>rating_score</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>09/27/2009</td>
      <td>appetites1</td>
      <td>Have all the toys in this car, have driven 700...</td>
      <td>smooth &amp; quiet</td>
      <td>0.8896</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>08/18/2009</td>
      <td>Joanne</td>
      <td>I've had my 2009 A5 for two months now, and I ...</td>
      <td>The lovely black glossy finish. The easy-to-us...</td>
      <td>0.9667</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>07/13/2009</td>
      <td>mgd99</td>
      <td>I held onto my 1998 A4 for 11 years, before ge...</td>
      <td>Exterior style (German with an Italian flair),...</td>
      <td>0.9853</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06/02/2009</td>
      <td>Aks</td>
      <td>Pros: Led lights, B&amp;O Sound System, Exterior, ...</td>
      <td>LED Lights, all wheel drive</td>
      <td>0.8725</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06/02/2009</td>
      <td>joe in ri</td>
      <td>Chose the A5 for its beauty and refinement. No...</td>
      <td>side assist, bluetooth and voice command, drop...</td>
      <td>0.7650</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>5</th>
      <td>06/01/2009</td>
      <td>Ted</td>
      <td>I am 6'5" tall and need a car I can fit in. Re...</td>
      <td>Interior and Exterior looks, quiet smooth ride...</td>
      <td>-0.2467</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>6</th>
      <td>05/20/2009</td>
      <td>Tommy</td>
      <td>After 3 months I still look back at it and smi...</td>
      <td>LIGHTS!!!! The sound, it's so clear - the ipod...</td>
      <td>0.9694</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>7</th>
      <td>04/27/2009</td>
      <td>SLINE</td>
      <td>It's fun, the roof needs to open, I mean reall...</td>
      <td>Styling, S-Line is the only way to go...</td>
      <td>-0.1658</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>8</th>
      <td>03/22/2009</td>
      <td>Laura</td>
      <td>The body of this car is by far the best lookin...</td>
      <td>Color and style are stunning. I absolutely lov...</td>
      <td>0.9878</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>9</th>
      <td>03/14/2009</td>
      <td>Drew</td>
      <td>After 8 months, I am still thrilled with the c...</td>
      <td>I-Pod function is seamless with the MMI contro...</td>
      <td>0.8689</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
</div>

