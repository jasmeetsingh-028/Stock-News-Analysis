from transformers import PegasusTokenizer, PegasusForConditionalGeneration  #tokenizer and model
from bs4 import BeautifulSoup #for scraping
from transformers import pipeline
import requests
import streamlit as st


from common import search_news_url, remove_unwanted_url, scrape_and_process, relevant_article, summarize, create_output_array



def main():

    st.title("Stock News Analysis Application")
    tickers = ['AAL', 'RBLX', 'F']
    selected_ticker = st.selectbox("Select a Ticker", tickers)

    st.write(f"Ticker Selected: {selected_ticker}")

    st.sidebar.image("https://imageio.forbes.com/specials-images/imageserve/617ab453e95e58ee7ce7de16/Stock-market-investment-and-anxious-future/960x0.jpg?height=473&width=711&fit=bounds", use_column_width=True)


    st.sidebar.header("Application Description")
    st.sidebar.write("This is a simple Streamlit application that allows you to analyze news related to a stock.")


    # Add your GitHub profile link
    st.sidebar.header("My GitHub Profile:")
    st.sidebar.write("[GitHub Profile](https://github.com/jasmeetsingh-028)")


    if st.button("Display Output"):
    
        raw_links = search_news_url(ticker= selected_ticker)

        exclude_keywords = ['maps', 'policies', 'prefrences', 'accounts', 'support']
        clean_urls = remove_unwanted_url(raw_links, exclude_keywords)
        # x = [len(clean_urls[t]) for t in tickers]
        # print(x)
        articles = scrape_and_process(clean_urls)
        #print(articles[0])

        relevant_articles = []
        relevant_urls = []
        for idx, art in enumerate(articles):
            if relevant_article(art, selected_ticker):
                #print(idx)
                relevant_articles.append(art)
                relevant_urls.append(clean_urls[idx])

        # print(relevant_urls)

        # if len(relevant_articles) == len(relevant_urls):
        #     print(True)

        model_name = "human-centered-summarization/financial-summarization-pegasus"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        summaries = summarize(relevant_articles, model = model, tokenizer = tokenizer)

        print(summaries)

        sentiment = pipeline('sentiment-analysis')
        scores = sentiment(summaries)

        print(scores)

        output = create_output_array(summaries, scores, relevant_urls, selected_ticker)

        print(output)

        for entry in output:
            st.write("Summary:", entry[1])
            st.write("Label:", entry[2])
            st.write("Score:", entry[3])
            st.write("URL:", entry[4])
            st.write("----")




if __name__=="__main__":
    main()
