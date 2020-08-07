# 2020T2-COMP9417-Project

## Overview
In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate."

In this project, we're presented with metadata on over 7,000 past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. 

## Acknowledgement

This dataset has been collected from TMDB. The movie details, credits and keywords have been collected from the TMDB Open API. This project uses the TMDB API but is not endorsed or certified by TMDB. Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows.

This is a playground prediction competition from Kaggle, link below:
> https://www.kaggle.com/c/tmdb-box-office-prediction/overview

## Dependencies
To run the programme correctly, you must make sure that you have installed these tools or files.

Python and its library.
- Python 3.7
- Numpy 1.19.1
- Pandas 1.1.0
- Matplotlib 3.3.0
- Scikit-learn 0.23.1
- Seaborn 0.10.1
- Catboost 0.23.2
- xgboost 1.1.1

>**Install using the command** `pip install -r  requirements.txt`

## What's included
- report.pdf
- data_preprocessing.py
- model.py
- final_train.csv
- final_test.csv
- submission.csv
- requirements.txt
- data.zip 
- appendix.zip 

## How to run 
You need to upzip data.zip before run the command:

`python data_preprocessing.py` 

`python model.py` 


## Teams
- Tim Luo `z5115679@ad.unsw.edu.au`
- Shu Yang `z5172181@ad.unsw.edu.au`
- YixiaoZhan `z5210796@ad.unsw.edu.au`
- Yue Qi `z5219951@ad.unsw.edu.au`

## Issues
The current modal for this project may not perfect. We are still exploring more suitable methods to improve the performance of outputs. Please contact us if there is anything wrong, appreciated.

## Maintainer
If you have any enquiries on this project or any other questions, please contact us at `z5115679@ad.unsw.edu.au`
