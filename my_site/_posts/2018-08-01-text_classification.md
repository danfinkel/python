---
title:        "Determining Movie Genre from its Review"
categories: 
    - TA
tags:
    - IMDB
    - Text
description:  "Combining Features to Classify Text"
header:
  teaser: /assets/images/wc_test.png
---

Text analysis is a continously growing application of data science that has many practical applications. At SMG we used data science to enhance many of our text analytic services we provide for clients. We overhauled our sentiment classification processes, developed highly accurate detectors for significant TA occurances, prototyped trend detectors and forecasting models for text and built tools and processes to extract entity variants from our large datasets of survey comments. 

For this exercise let's understand how much information about a movie genre is captured in review content. The code used to generate the content shown in this post can be found [here](https://github.com/danfinkel/python/tree/master/text_classification).

## The Stanford IMDB Dataset

The AI lab at Stanford ([SAIL](http://ai.stanford.edu)) has done a lot cool stuff over the the years and provides a lot of services to the data science community (the [GloVe word embeddings data set](https://nlp.stanford.edu/projects/glove/) is just one good example).

For this exercise we are going to use the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) developed and hosted by the Stanford AI group. The dataset consists of 50,000 movie reviews that have been labeled positive and negative. Half of the reviews are labeled positive while half are labeled negative. The data is organized such that half the reviews are in a test folder and half in a training folder (so 12,500 reviews are positive-training, 12,500 are positive-negative, etc...). You can see the associated paper [here](http://www.aclweb.org/anthology/P11-1015).
 
## Scraping Genre from IMDB

The Stanford review dataset has very little meta-data about the film being reviewed -- the only information provided is a url to the films corresponding IMDB page. Fortunately in most cases that page has the films genre embedded in it. The code snippet below is used to extract the genre(s) of the film from its IMDB page.

```python
        # request the html data
        # need to remove excess url stuff
        response = requests.get(row[1]['url'].split('/usercomments')[0])

        # parse the text
        soup = BeautifulSoup(response.text, 'html.parser')

        # genre data is in the script tag
        scriptrows = soup.find_all('script')
        try:
            # look in the first one
            cand = [r for r in scriptrows if str(r).split('genre')[0] != str(r)][0]    
            try:
                # multiple genres
                genres = ast.literal_eval(str(cand).split('genre')[1].split(']')[0].replace("\n","").replace('": [',"").lstrip().rstrip())
            except SyntaxError:
                # hit an error
                # because only one genre
                genres = (str(cand).split('genre')[1].split(',')[0].replace("\n","").replace('":',"").lstrip().rstrip().replace('"',""))

            # grab the movie title
            movie_title = soup.title.text.replace(' - IMDb', "")

            # record the data
            movie_data.append((row[0], movie_title, genres))
        except:
            # no genres found -- skip this entry
            print "Error with %i" % row[0]
```

The result is a dataframe indexed to the dataset that contains the title and genres of each reviewed film:

| ref | title | genres |
|-----|-------|--------|
| 0 | Futz (1969) | (Comedy) |
| 1 | Stanley & Iris (1990) |	(Drama, Romance) |
| 2 | Stanley & Iris (1990) | (Drama, Romance) |
| 3 | Stanley & Iris (1990) |	(Drama, Romance) |
| 4 | The Mad Magician (1954) |(Horror, Mystery, Thriller) |

You can see a couple of attributes stand out about the data. First, there are multiple reviews for the same movie. We can see this if we dig into the data. Here is the review for ref 2:

> I saw the capsule comment said "great acting." In my opinion, these are two great actors giving horrible performances, and with zero chemistry with one another, for a great director in his all-time worst effort. Robert De Niro has to be the most ingenious and insightful illiterate of all time. Jane Fonda's performance uncomfortably drifts all over the map as she clearly has no handle on this character, mostly because the character is so poorly written. Molasses-like would be too swift an adjective for this film's excruciating pacing. Although the film's intent is to be an uplifting story of curing illiteracy, watching it is a true "bummer." I give it 1 out of 10, truly one of the worst 20 movies for its budget level that I have ever seen.

And here is the review for ref 1:
> Robert DeNiro plays the most unbelievably intelligent illiterate of all time. This movie is so wasteful of talent, it is truly disgusting. The script is unbelievable. The dialog is unbelievable. Jane Fonda's character is a caricature of herself, and not a funny one. The movie moves at a snail's pace, is photographed in an ill-advised manner, and is insufferably preachy. It also plugs in every cliche in the book. Swoozie Kurtz is excellent in a supporting role, but so what?<br /><br />Equally annoying is this new IMDB rule of requiring ten lines for every review. When a movie is this worthless, it doesn't require ten lines of text to let other readers know that it is a waste of time and tape. Avoid this movie.

Digging a little further we can count the number of unique movies in each data set:

```python
len(df_postest_genres['title'].unique())
```

| Dataset | Polarity | Unique Movies |
|---------|----------|---------------|
| Train   | Positive | 1,382         |
| Train   | Negative | 2,927         |
| Test    | Positive | 1,344         |
| Test    | Negative | 2,982         |

The genres themselves also have interesting attributes. The first chart shows raw counts of assigned movie genres across the 2,927 movies in the negative-reviews training folder.

![no-alignment](/assets/images/genre_breakout_neg_training_set.png)

There are a lot of dramas and comedies in the dataset, some romance, action and sci-Fi movies and only a few Westerns, Documentaries and Sports-related films.

Movies in the dataset can also have multiple genres attached to them. For example, according to IMDB, *Stanley & Iris* is classified as both a drama and a romance. 

![image-left](/assets/images/Drama_Romance.png){: .align-left} IMDB-labeled movie genres overlap in a very intuitive way. For example, as shown to the left, a significant number of Romance movies are also Dramas. The converse is not as strong -- i.e., there are a lot of drama movies that are not considered romances. This also seems intuitive -- dramas can be much more than just romances.

We can visualize the relationships between co-occurances of different genres with a clustermap

![no-alignment](/assets/images/clustermap_negative_train.png)

Once again the results are somewhat intuitive. The "Drama" genre co-occurs with many different smaller genres (strong co-occurances especially occur with "Sport" and "War") while there are a few strong relationships to be seen ("News" and "Talk Show" for example).

We can use this data to help us design the proper classification problem. 

![image-left](/assets/images/Drama_Comedy.png){: .align-left} From the bar graph above we know there is plenty of reviews for "Drama" and "Comedy" movies. We can also see that their co-occurance isn't overwhelming, i.e., only 277 movies in the negative set are labeled both "Comedy" and "Drama". This provides a good starting problem -- how well can we train a classifier to detect if a movie is a "Comedy" or a "Drama"?



## Building an NLP Classifier

## Performance Results

Paragraphs are separated by a blank line.

2nd paragraph. *Italic*, **bold**, and `monospace`. Itemized lists
look like:

  * this one
  * that one
  * the other one

