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

