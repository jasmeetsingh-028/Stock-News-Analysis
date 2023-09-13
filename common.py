from bs4 import BeautifulSoup
import requests
import re

def search_news_url(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    a_tags = soup.find_all('a')  #a tags represent links - list of a tags
    hrefs = [link['href'] for link in a_tags]
    return hrefs

def remove_unwanted_url(urls, exclude_list):
    val = []
    for url in urls: 
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0] #getting specific url component
            val.append(res)
    return list(set(val))

def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

def relevant_article(article, ticker):
    relevant_keys = {'RBLX': 'roblox', 'F': 'Ford Motor', 'AAL': 'American Airlines'}
    if ticker in article or relevant_keys[ticker] in article.lower():
        return True
    else:   
        return False
    
def summarize(articles, model, tokenizer):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def create_output_array(summaries, scores, urls, ticker):
    output = []
    for counter in range(len(summaries)):
        output_this = [
            ticker,
            summaries[counter],
            scores[counter]['label'],
            scores[counter]['score'],
            urls[counter]
        ]
        output.append(output_this)
    return output