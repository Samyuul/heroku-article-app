# pip install dash==2.1.0
# pip install dash-bootstrap-components==1.0.3
# pip install requests==2.25.1
# pip install bs4==0.0.1
# pip install nltk=3.6.1
# pip install numpy==1.20.1
# pip install pandas==1.2.4
# pip install sklearn
# pip install lxml==4.6.3
# pip install gunicorn==20.0.4

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import requests
from bs4 import BeautifulSoup

import nltk

import re
import numpy as np
import pandas as pd

import time
import pickle

from nltk.stem import WordNetLemmatizer

# Global Variables
# Global storage of results
headline_list = [[], [], [], []]
url_list = [[], [], [], []]
article_list = [[], [], [], []]
summary_list = [[], [], [], []]
keyword_list = [[], [], [], []]
occurrence_list = [[], [], [], []]
class_list = [[], [], [], []]

news_select = 0  # Index to select desired news site

# Text for markdowns for instructions
markdown_text1 = '''

Please select a news site below out of **CBC**, **The Guardian**, **NBC**, or **CBS** to view articles from.

This process will take 15 to 20 seconds depending on your internet speed. 

'''

markdown_text2 = '''

Afterwards, please select a article based on the headline in the dropdown below. 
You will be provided a summarized version, alongside a histogram of the most common keywords found.

Furthermore, you can continue on the full article by pressing **Read More**.

'''

markdown_text3 = '''

Finally, you can filter the list of articles by the tags via the dropdown menu below.

'''

# Load Stopwords
stop_file = open('stopwords-en.txt', 'r')
stop_words = stop_file.read().split('\n')

# Load Word Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load Trained TFIDF Vectorizer
filename = 'tfidf.pickle'
with open(filename, 'rb') as data:
    tfidf_ = pickle.load(data)

# Load Trained SVM model
loaded_model = pickle.load(open('SVM.pickle', 'rb'))


# Functions
def summarizeText(text, percent=0.5):
    # Generate a text summary based on an extractive algorithm by creating histogram of word occurrences

    # Inputs:  - text (String): Unedited source text
    #          - percent (Float): Percentage of word score to retain in summary

    # Output:  - (List of Strings): A list of most important sentences in chronological order

    # Generate histogram for word occurrence
    clean_text = text
    clean_text = re.sub(r'\[[0-9]*]', ' ', clean_text)  # Replace in-text references (ex. [0]) with empty space
    clean_text = clean_text.lower()  # Convert to lower case

    clean_text = re.sub(r"[^a-zA-Z0-9]", " ", clean_text)  # Replace special characters with empty space
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with a single space

    # Dictionary for histogram
    word_count = {}

    # Loop through tokenized words without stop words, generate dictionary for histogram
    for word in nltk.word_tokenize(clean_text):

        if word not in stop_words:  # Not stopwords

            # Increment word count
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1

    # create empty dictionary to house sentence score
    sentence_score = {}

    sentences = nltk.sent_tokenize(text)  # Separate text into words

    # Loop tokenized sentences and calculated score
    for sentence in sentences:

        for word in nltk.word_tokenize(sentence.lower()):  # Loop through each word

            if word in word_count.keys():  # Word is in word histogram

                # Only sentences less than 50 words
                if len(sentence.split(' ')) < 50:

                    # Keep track of sentence score
                    if sentence not in sentence_score.keys():
                        sentence_score[sentence] = word_count[word]
                    else:
                        sentence_score[sentence] += word_count[word]

    # Generate list of keywords and their corresponding number of occurrences
    word_count_table = pd.DataFrame.from_dict(word_count, orient='index').rename(columns={0: 'score'})
    word_count_table = word_count_table.sort_values(by='score', ascending=False).head(7)
    keyword_list = word_count_table.index.tolist()
    value_list = word_count_table['score'].tolist()

    # Generate dataframe based on dictionary
    df = pd.DataFrame(
        {"Text": sentence_score.keys(), "Score": sentence_score.values()}
    )

    # Sort the dictionary based on score
    df = df.sort_values(by=['Score'], ascending=False)
    total_score = df.sum().Score
    curr_score = 0
    count = 0

    # Count how many sentences are required to match the desired percentage of score
    for i, row in df.iterrows():

        curr_score += row.Score
        count += 1

        if curr_score > total_score * percent:  # Enough sentences noted
            break

    # Generate the text based on text chronological order
    summary = df.head(count)
    summary = summary.sort_index()
    summary_text = ""

    # Append text together separated by newline characters
    for i, row in summary.iterrows():
        summary_text += "\n\n" + row.Text

    return summary_text[2:], keyword_list, value_list


def generateGuardianHeadlines():
    # Get all headlines on front page of the guardian along with URL links
    # Input:  - None
    # Output: - (List of tuples): Contains headlines and URLs as (Headline, URL)

    html_text = requests.get('https://www.theguardian.com/international').text
    soup = BeautifulSoup(html_text, 'lxml')

    # Find all possible headline hyperlink elements
    headlines = soup.find_all('a', class_='u-faux-block-link__overlay js-headline-text')[0:25]  # Limit to first 25

    articles = []
    for headline in headlines:

        entry = (headline.text, headline['href'])

        if entry not in articles:  # Avoid duplicates
            articles.append(entry)

    return articles


def getArticleTextGuardian(url):
    # Return article text based on specified URL
    # Input:  - url (String): URL of desired guardian article
    # Output: - (String): All text found in article body

    # Get HTML of article via URL
    article_text = requests.get(url).text
    article_html = BeautifulSoup(article_text, 'lxml')

    # Get class code for article text
    body = article_html.find('div', class_='article-body-commercial-selector')
    code = body.find('p')['class']

    # Find all paragraph elements
    all_text_containers = body.find_all('p', class_=code)

    # Append text together
    total_text = ""
    for text_container in all_text_containers:
        total_text += " " + text_container.text

    return total_text[1:]


def getAllArticleTextGuardian():
    # Partition data into four lists of headlines, URLs, article body, and predicted article class from the Guardian

    # Input:  - None

    # Output: - (List of Strings): List of headlines
    #         - (List of Strings): List of URLs
    #         - (List of Strings): List of article texts
    #         - (List of Integers): predicted article class

    # Generate all possible URLs from home page
    articles = generateGuardianHeadlines()
    headline_list = []
    url_list = []
    text_list = []
    class_list = []

    # loop for each article
    for article in articles:

        try:  # See if there is an article text body
            curr_text = getArticleTextGuardian(article[1])

            if curr_text is not None:  # Non-empty text returned
                headline_list.append(article[0])
                url_list.append(article[1])
                text_list.append(curr_text)
                class_list.append(predictClass(curr_text))  # Predict Using SVM Model

        except (Exception, ):  # No text body found
            pass

        time.sleep(0.25)  # Limit request rate

    return headline_list, url_list, text_list, class_list


def generateCBCHeadlines():
    # Get all headlines on front page of the CBC along with URL links
    # Input:  - None
    # Output: - (List of tuples) Contains headlines and URLs as (Headline, URL)

    html_text = requests.get('https://www.cbc.ca/news').text
    soup = BeautifulSoup(html_text, 'lxml')

    articles = []
    headlines = soup.find_all('a', class_='cardText')

    # One type of headlines
    for headline in headlines:

        url = 'https://www.cbc.ca' + headline['href']
        texts = headline.find('h3').text

        if 'photo' not in url and 'player' not in url and 'radio' not in url:  # Only add articles
            if (texts, url) not in articles:  # Avoid duplicates
                articles.append((texts, url))

    # Another possible headline
    headlines = soup.find_all('a', class_='cardListing')

    for headline in headlines:
        articles.append((headline.find('h3', class_='headline').text, 'https://www.cbc.ca' + headline['href']))

    return articles


def getArticleTextCBC(url):
    # Return article text based on specified URL
    # Input:  - url (String): URL of desired guardian article
    # Output: - (String): All text found in article body

    # Get HTML of article via URL
    article_text = requests.get(url).text
    article_html = BeautifulSoup(article_text, 'lxml')

    # Search for all paragraph elements in article text body
    article_body = article_html.find('div', class_='story')
    sentences = article_body.find_all('p')

    total_text = ""
    for sentence in sentences:

        if '<br>' not in sentence.text and 'href=' not in sentence.text:
            total_text += " " + sentence.text

    return total_text[1:]


def getAllArticleTextCBC():
    # Partition data into four lists of headlines, URLs, article body, and predicted article class from CBC

    # Input:  - None

    # Output: - (List of Strings): List of headlines
    #         - (List of Strings): List of URLs
    #         - (List of Strings): List of article texts
    #         - (List of Integers): predicted article class

    # Generate all possible URLs from home page
    articles = generateCBCHeadlines()
    headline_list = []
    url_list = []
    text_list = []
    class_list = []

    # loop for each article
    for article in articles:

        try:  # See if there is an article text body
            curr_text = getArticleTextCBC(article[1])

            if curr_text is not None:  # Non-empty text returned
                headline_list.append(article[0])
                url_list.append(article[1])
                text_list.append(curr_text)
                class_list.append(predictClass(curr_text))  # Predict Using SVM Model

        except (Exception, ):  # Ignore the request
            pass

        time.sleep(0.25)  # Limit request rate

    return headline_list, url_list, text_list, class_list


def generateNBCHeadlines():
    # Get all headlines on front page of NBC along with URL links
    # Input:  - None
    # Output: - (List of tuples): Contains headlines and URLs as (Headline, URL)

    html_text = requests.get('https://www.nbcnews.com/').text
    soup = BeautifulSoup(html_text, 'lxml')

    # Find all hyperlinks
    articles = []
    headlines = soup.find_all('li', class_='styles_item__sANtw')

    # Loop for each possible headline
    for headline in headlines:

        try:
            curr = headline.find_all('a')[-1]
            url = curr['href']
            text = curr.text
        except (Exception,):
            pass

        # Ignore videos
        if 'video' not in url:
            if (text, url) not in articles:  # Avoid duplicates
                articles.append((text, url))

    return articles


def getArticleTextNBC(url):
    # Return article text based on specified URL from NBC
    # Input:  - url (String): URL of desired NBC article
    # Output: - (String): All text found in article body

    # Get HTML of article via URL
    article_text = requests.get(url).text
    article_html = BeautifulSoup(article_text, 'lxml')
    article_body = article_html.find('div', class_='article-body__content')
    sentences = article_body.find_all('p')

    total_text = ""
    for sentence in sentences:
        total_text += " " + sentence.text

    return total_text[1:]


def getAllArticleTextNBC():
    # Partition data into four lists of headlines, URLs, article body, and predicted article class from NBC

    # Input:  - None

    # Output: - (List of Strings): List of headlines
    #         - (List of Strings): List of URLs
    #         - (List of Strings): List of article texts
    #         - (List of Integers): predicted article class

    # Generate all possible URLs from home page
    articles = generateNBCHeadlines()
    headline_list = []
    url_list = []
    text_list = []
    class_list = []

    # loop for each article
    for article in articles:

        try:  # See if there is an article text body
            curr_text = getArticleTextNBC(article[1])

            if curr_text is not None:  # Non-empty text returned
                headline_list.append(article[0])
                url_list.append(article[1])
                text_list.append(curr_text)
                class_list.append(predictClass(curr_text))  # Predict Using SVM Model

        except (Exception, ):  # Ignore the request
            pass

        time.sleep(0.25)  # Limit request rate

    return headline_list, url_list, text_list, class_list


def generateCBSHeadlines():
    # Get all headlines on front page of CBS along with URL links
    # Input:  - None
    # Output: - (List of tuples): Contains headlines and URLs as (Headline, URL)

    html_text = requests.get('https://www.cbsnews.com/').text
    soup = BeautifulSoup(html_text, 'lxml')

    # Find all hyperlinks
    headlines = soup.find_all('section', class_='list-grid--with-hero')

    articles = []

    # Loop for each block of hyperlinks
    for headline in headlines:
        curr_block = headline.find_all('a', class_='item__anchor')

        for curr in curr_block:  # Loop for each hyperlink

            text = curr.find('h4', class_='item__hed').text.strip()
            url = curr['href']

            if (text, url) not in articles:
                articles.append((text, url))

    return articles


def getArticleTextCBS(url):
    # Return article text based on specified URL from CBS
    # Input:  - url (String): URL of desired CBS article
    # Output: - (String): All text found in article body

    # Get HTML of article via URL
    article_text = requests.get(url).text
    article_html = BeautifulSoup(article_text, 'lxml')
    article_body = article_html.find('section', class_='content__body')
    sentences = article_body.find_all('p')

    total_text = ""
    for sentence in sentences:
        total_text += " " + sentence.text

    return total_text[1:]


def getAllArticleTextCBS():
    # Partition data into four lists of headlines, URLs, article body, and predicted article class from CBS

    # Input:  - None

    # Output: - (List of Strings): List of headlines
    #         - (List of Strings): List of URLs
    #         - (List of Strings): List of article texts
    #         - (List of Integers): predicted article class

    # Generate all possible URLs from home page
    articles = generateCBSHeadlines()
    headline_list = []
    url_list = []
    text_list = []
    class_list = []

    # loop for each article
    for article in articles:

        try:  # See if there is an article text body
            curr_text = getArticleTextCBS(article[1])

            if curr_text is not None:  # Non-empty text returned
                headline_list.append(article[0])
                url_list.append(article[1])
                text_list.append(curr_text)
                class_list.append(predictClass(curr_text))  # Predict Using SVM Model

        except (Exception, ):  # Ignore the request
            pass

        time.sleep(0.25)  # Limit request rate

    return headline_list, url_list, text_list, class_list


def predictClass(text):
    # Predict article class by applying an SVM on it's TFIDF feature

    # Input:  - text (String): Source text of article

    # Output: - (Integer): Output that corresponds to the predicted class
    #           (0 = Business, 1 = Entertainment, 2 = Politics, 3 = Sports, 4 = Tech, 5 = Uncategorized)

    text = text.replace('\n', ' ')  # New line
    text = text.replace('\r', ' ')  # New line
    text = text.replace('    ', ' ')  # Tab
    text = text.replace('"', '')  # Quotes
    text = text.lower()  # Lowercase

    # Remove possessive forms
    text = text.replace("'s", '')

    # Remove all other special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    text = " ".join(text.split())  # Remove multiple spaces

    curr_text_words = text.split(" ")  # Tokenize

    lemma_list = []

    # Lemmatize each word
    for word in curr_text_words:

        if word not in stop_words:  # Not in stop words list
            lemma_list.append(lemmatizer.lemmatize(word, pos='v'))

    result = " ".join(lemma_list)

    # Tfidf and probability predictions
    pred = loaded_model.predict_proba(tfidf_.transform([result]).toarray())[0]

    # Enough probability to classify
    if pred[np.argmax(pred)] > 0.65:
        return np.argmax(pred)
    else:  # Return unclassified
        return 5


# Function Dictionary for article scrapping
get_func = {0: getAllArticleTextCBC, 1: getAllArticleTextGuardian, 2: getAllArticleTextNBC, 3: getAllArticleTextCBS}

# App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# HTML Structure
app.layout = html.Div([
    dbc.Row(  # Title
        dbc.Col(
            html.H1(
                "Article Scraper and Summarizer",
                className='text-center',
                style={'marginTop': '1vh', 'font-size': '4vh', 'color': 'white', 'marginBottom': '1vh'}
            ),
        ), style={'width': '100%', 'height': '6vh', 'backgroundColor': '#212529'}
    ),

    dbc.Row([  # App body
        dbc.Col([  # Article selector
            html.Div([
                dcc.Markdown(  # Instructions
                    children=markdown_text1,
                    id='instruction',
                    style={'font-size': '1.6vh'}
                ),

                dcc.Dropdown(  # Dropdown selector for news site
                    options=[
                        {'label': 'CBC', 'value': 0},
                        {'label': 'The Guardian', 'value': 1},
                        {'label': 'NBC', 'value': 2},
                        {'label': 'CBS', 'value': 3}
                    ],
                    id='news_select',
                    style={'font-size': '1.6vh', 'color': 'white'}
                ),

                dcc.Markdown(  # Instructions
                    children=markdown_text2,
                    id='instruction2',
                    style={'marginTop': '1vh', 'marginBottom': '1vh', 'font-size': '1.6vh'}
                ),

                dcc.Dropdown(  # Dropdown selector for news site
                    id='article_select',
                    options=[],
                    style={'font-size': '1.6vh', 'color': 'white'}
                ),

                html.Div(
                    html.Div(
                        dcc.Graph(  # Word occurrence histogram
                            figure={
                                'data': [
                                    {'x': ["", "", "", "", "", "", ""], 'y': [0, 0, 0, 0, 0, 0, 0], 'type': 'bar',
                                     'name': 'SF'},
                                ],
                                'layout': {  # Color of chart
                                    'plot_bgcolor': '#212529',
                                    'paper_bgcolor': '#212529',
                                    'font': {
                                        'color': 'white'
                                    },
                                    'margin': {  # Reduce margins
                                        'r': '25',
                                        'l': '25',
                                        'b': '25',
                                        't': '45'
                                    },
                                    'title': 'Keywords in the Article'  # Title of chart
                                },
                            },

                            style={'width': "95%", 'height': "100%", 'margin-left': 'auto', 'margin-right': 'auto',
                                   'marginTop': '2vh', 'marginBottom': '1vh'},
                            id='keyword-graph'
                        ),

                        style={'height': '40vh'}
                    ),
                ),

                dcc.Markdown(  # Instructions
                    children=markdown_text3,
                    id='instruction3',
                    style={'font-size': '1.6vh'}
                ),

                dcc.Dropdown(  # Filter for article class
                    options=[
                        {'label': 'Business', 'value': 0},
                        {'label': 'Entertainment', 'value': 1},
                        {'label': 'Politics', 'value': 2},
                        {'label': 'Sports', 'value': 3},
                        {'label': 'Tech', 'value': 4},
                        {'label': 'Uncategorized', 'value': 5}
                    ],
                    value=[0, 1, 2, 3, 4, 5],
                    multi=True,
                    id='tag-select',
                    disabled=True
                )],

                style={'marginLeft': '2vh', 'marginTop': '2vh', 'marginBottom': '2vh', 'height': '95%',
                       'overflow': 'auto', 'backgroundColor': '#212529', 'color': 'white'}
            )],

            style={'height': '100%', 'backgroundColor': '#212529'}
        ),

        dbc.Col(
            html.Div([
                html.Div([
                    dcc.Textarea(  # Summarized Article body
                        id='article-text',
                        value="",
                        style={'width': '100%', 'height': '100%', 'resize': 'none', 'color': 'black',
                               'backgroundColor': '#b3b3b3', 'border': 'none'},
                        disabled='True',
                        readOnly=True,
                        draggable=False
                    )],

                    id='textBox',
                    style={'height': '84vh', 'marginBottom': '1vh'}
                ),

                # Read more button
                html.A("Read More", href='', target="_blank", id='hyperlink', hidden=True,
                       style={'text-decoration': 'none', 'color': '#bdeeff'})]
                ),

            style={'height': '97%', 'marginTop': '2vh', 'width': '60%'},

        )],

        style={'width': '100%', 'height': '94vh', 'backgroundColor': '#212529'}

    ),

    dcc.Loading(  # Loading screen during scraping
        id="loading-1",
        children=[html.Div([html.Div(id="loading-output-1")])],
        type='default',
        style={'backgroundColor': '#212529', 'opacity': '0.6'},
        fullscreen=True
    )],

    style={'width': '100%', 'height': '100vh', 'backgroundColor': '#212529'}
)


# Callbacks for updating the application
# Generate the article list
@app.callback(
    Output('tag-select', 'value'),
    Output('loading-output-1', 'children'),
    Input('news_select', 'value'), prevent_initial_call=True
)
def update_article_selector(news_site):
    # Generates the list of headlines, URLs, article body, and predicted article class for a specific news site

    # Input: - news_site (Integer): Index that controls current selected news site:
    #          (0 = CBC, 1 = The Guardian, 2 = NBC, 3 = CBS)

    # Output: - (List of Integers): Controls selected filter tags (Defaults to Select All)
    #         - (String): Empty output in order to update children of loading component to induce loading screen

    if news_site is not None:  # No news site selected

        headline_list[news_site], url_list[news_site], article_list[news_site], class_list[news_site] = get_func[
            news_site]()
        summary_list[news_site] = [None] * len(headline_list[news_site])
        keyword_list[news_select] = [None] * len(headline_list[news_site])
        occurrence_list[news_select] = [None] * len(headline_list[news_site])

    return [0, 1, 2, 3, 4, 5], ""


@app.callback(
    Output('article_select', 'options'),
    Output('article_select', 'value'),
    Input('tag-select', 'value'),
    [State('news_select', 'value')], prevent_initial_call=True
)
def filter_list(tags_select, news_site):
    # Generate a filtered list based on currently selected news site and desired filter tags

    # Input: - tags_select (List of Integers): A list containing desired tags corresponding to:
    #          (0 = Business, 1 = Entertainment, 2 = Politics, 3 = Sports, 4 = Tech, 5 = Uncategorized)
    #        - news_site (Integer): Index that controls current selected news site:
    #          (0 = CBC, 1 = The Guardian, 2 = NBC, 3 = CBS)

    # Output: - (List of tuples): Contains possible article headlines and corresponding index in the total list as
    #           (headline, index)
    #         - (Integer): Index of selected article (-1 defaults to no selection)

    filtered_list = []

    if news_site is not None:  # No news site selected

        for i, headline in enumerate(headline_list[news_site]):
            if class_list[news_site][i] in tags_select:
                filtered_list.append({'label': headline, 'value': i})

    return filtered_list, -1


@app.callback(
    Output('article-text', 'value'),
    Output('keyword-graph', 'figure'),
    Output('hyperlink', 'href'),
    Output('hyperlink', 'hidden'),
    Output('tag-select', 'disabled'),
    Input('article_select', 'value'),
    [State('news_select', 'value')], prevent_initial_call=True
)
def update_article_text(index, news_site):
    # Update the text area to display the desired summarized article alongside a word histogram

    # Input: - index (Integer): Index of desired article body in the total article list
    #        - news_site (Integer): Index that controls current selected news site:
    #        (0 = CBC, 1 = The Guardian, 2 = NBC, 3 = CBS)

    # Output - (String): Summarized article text
    #        - (Graph): Word occurrence histogram
    #        - (String): Hyperlink to current article
    #        - (Boolean): Used to hide / un-hide read more button
    #        - (Boolean): Used to disable / un-disable filter tags

    if index is not None and index != -1:  # Display new articles

        summary_list[news_site][index], keyword_list[news_select][index], occurrence_list[news_select][
            index] = summarizeText(article_list[news_site][index])

        figure = {
            'data': [
                {'x': keyword_list[news_select][index], 'y': occurrence_list[news_select][index], 'type': 'bar',
                 'name': 'SF'},
            ],
            'layout': {  # Color of chart
                'plot_bgcolor': '#212529',
                'paper_bgcolor': '#212529',
                'font': {
                    'color': 'white'
                },
                'margin': {  # Reduce margins
                    'r': '25',
                    'l': '25',
                    'b': '25',
                    't': '45'
                },
                'title': 'Keywords in the Article'  # Title of chart
            }
        }

        return summary_list[news_site][index], figure, url_list[news_site][index], False, False

    else:  # Reset everything otherwise

        figure = {  # Empty bar chart
            'data': [
                {'x': ["", "", "", "", "", "", ""], 'y': [0, 0, 0, 0, 0, 0, 0], 'type': 'bar', 'name': 'SF'},
            ],
            'layout': {  # Color of chart
                'plot_bgcolor': '#212529',
                'paper_bgcolor': '#212529',
                'font': {
                    'color': 'white'
                },
                'margin': {  # Reduce margins
                    'r': '25',
                    'l': '25',
                    'b': '25',
                    't': '45'
                },
                'title': 'Keywords in the Article'  # Title of chart
            },
        }

        return "", figure, "", True, False


if __name__ == '__main__':
    app.run_server(debug=False)
