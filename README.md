---
title: "Sentiment analysis of video games reviews on Amazon"
author: "Miguel Almodôvar"
date: "15/06/2022"
output:
  html_document: default
  pdf_document: default
---

## Data loading and processing

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textmodels)
library(quanteda.textstats)
library(dplyr)
library(textplot)
library(ggplot2)
library(caret)
```

Load the dataset

```{r}
reviews <- read.csv("https://drive.google.com/uc?id=1BAsGEwOxWCz0OJQZ6Z5UE_IxrLr2xKvm&export=download&confirm=t&uuid=1ebf996a-18b1-4866-ae3f-1ccee0a120d1", sep=";")
reviews <- reviews %>% select(reviewText, overall,summary)
reviews$ID <- 1:nrow(reviews)
summary(reviews)
```

Create column where overall 5 & 4 = 'good', overall 3 = 'med' and overall 2 & 1 = 'bad'

```{r}
reviews <- subset(reviews, overall != 3)
reviews <- reviews %>% mutate(sentiment =
                     case_when(overall <= 2 ~ "bad", 
                               overall >= 4 ~ "good")
)
```

```{r}
sentimentable <- table(reviews$sentiment)
labels <- paste(names(sentimentable), "\n", sentimentable, sep="")
pie(sentimentable, labels = labels,
   main="Pie Chart of sentimenting distribution")
```

We can observe that there are far more good reviews than bad. We will randomly sample 20000 of each category for further analysis and modelling

```{r}

goodsample <- reviews[ sample(which ( reviews$sentiment == "good" ) ,20000), ]
badsample <- reviews[ sample(which ( reviews$sentiment == "bad" ) ,20000), ]
reviews <- rbind(goodsample,badsample)
summary(reviews)
```

## Processing text and creating corpus

Create corpus and remove stopwords, only keeping some important ones that can indicate sentiment

```{r}
custom_english <- stopwords("english")[! stopwords("english") %in% c("isn't",      "aren't",     "wasn't" ,    "weren't"  ,  "hasn't"  ,   "haven't"  ,  "hadn't"  ,   "doesn't",    "don't" ,     "didn't","won't","wouldn't","shan't","shouldn't","can't","cannot","couldn't",   "mustn't","no","nor","not")]
corp <- corpus(reviews, text_field = 'reviewText')
dtm <- corp |>
  tokens(remove_punct = T, remove_numbers = T, remove_symbols = T) |>   
  tokens_tolower() |>                                                    
  tokens_remove(custom_english) |>                                     
  tokens_wordstem() |>
  dfm()

```

Word cloud for 'good' reviews

```{r}
textplot_wordcloud(dfm_subset(dtm, sentiment == "good"),max_words=100,rotation=0,min_size=1) 
```

```{r}
tstat_freq <- textstat_frequency(dfm_subset(dtm, sentiment == "good"), n =10)
ggp <- ggplot(tstat_freq, aes(reorder(feature,frequency), frequency)) +   
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal() + ggtitle("Top 10 words by frequency for 'good' reviews")
ggp

```

Word cloud for 'bad' reviews

```{r}
textplot_wordcloud(dfm_subset(dtm, sentiment == "bad"),max_words=100,rotation=0,min_size=1) 
```

```{r}
tstat_freq <- textstat_frequency(dfm_subset(dtm, sentiment == "bad"), n =10)
ggp <- ggplot(tstat_freq, aes(reorder(feature,frequency), frequency)) +   
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()+ ggtitle("Top 10 words by frequency for 'bad' reviews")
ggp
```

## Building the model and testing

Get the training and testing sets (80% for training 20% for testing)

```{r}

set.seed(300)
id_train <- createDataPartition(docvars(dtm)$sentiment, p = .8, 
                                  list = FALSE, 
                                  times = 1)
```

```{r}
# get training set
dfmat_test <- dfm_subset(dtm, docvars(dtm)$ID %in% id_train)

# get test set (documents not in id_train)
dfmat_training <- dfm_subset(dtm, !docvars(dtm)$ID %in% id_train)
```

We'll be using the naive bayes model to classify the documents. From the training summary we can see some feature scores and notice that positive words like 'love' have a higher score for 'good' and negative words like 'hate' have higher score for 'bad'.

```{r}
tmod_nb <- textmodel_nb(dfmat_training, docvars(dfmat_training)$sentiment)
summary(tmod_nb)
```

Now we test our model. We can see that the accuracy we got was 82% which is acceptable for a model of this size. Its important to note that Naive Bayes assumes predictors as being totally independent from each other, and that never happens in language. However due to its speed, this algorithm is still widely used for spam detection and sentiment analysis.

```{r}
dfmat_matched <- dfm_match(dfmat_test, features = featnames(dfmat_training))
actual_class <- docvars(dfmat_matched)$sentiment
predicted_class <- predict(tmod_nb, newdata = dfmat_matched)
tab_class <- table(actual_class, predicted_class)
confusionMatrix(tab_class, mode = "everything",positive = 'good')
confusionMatrix
```
