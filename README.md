# Chatbot
The project is built for people living in isolation or looking for a friend. The project uses Natural language Processing technology to interpret the user's msg and give a response. This project is benefetial for all those who want's to talk to someone and share their feelings. The chatbot is built as a web API using flask as backend. 
![video_description](/video.gif)

# Data Acquisition:
The data that I used in the chatbot is custom-developed by me. I used a JSON file format to store the data, then load it into the python file using the JSON module. 
The json file has three main components.
<ul>
<li> Tag
<li> Patterns
<li> Responses
  </ul>
  
The "Tag" represents the specific tag or label associated.
The "Patterns" represents the type of text that may encounter related to the specific tag or label.
The "Response" represents the responses that the bot will give when that specific tag is encountered.
Together all three components enable the chatbot to analyze the sentiment of the text and give an appropriate response related to it.

# Data Preparation:
After loading the json file into our project using the "json. load()" function, we need to prepare our data to organize the data in a format to train our model. Here we are organizing the data to form a pandas dataframe. To do this, we'll store each of our patterns and its respective tag into a list then create a dataframe using pd.DataFrame() function. Our dataframe will look like this.
![top_5_rows](/data.jpg)

We are also creating a dictionary to store respective responses related to the tags.

Now we have our data loaded, the next step is to do some preprocessing.
First, we'll convert all our sentences to lower case and remove all the punctuation from the sentences. 
Since the model is trained only on integer data, we'll apply tokenization to our dataset. Tokenization is a process of assigning a word with an integer so that our sentence may be deduced in a numeric format. We'll set the max limit of these words to 2000.
In our dataset, we have sentences of different lengths of words, so we need to apply "padding" to our dataset, which means standardize our rows to a similar length. Padding is the process of adding zeros in front of all those rows having less fewer integers(words).
Lastly, we'll also apply label encoding to our tags as well using the sklearn preprocessing LabelEncoding() function.
