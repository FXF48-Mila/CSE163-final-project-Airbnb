# Instruction for running the project
## Introduction from the author:
This project has a total of four coding files, which are `final.py`, `data_cleaning.py`, `test.py`, and `testing.r`. Among them, the main file is the `final.py`, which contains all the code needed for the reaserch questions and __is the one that you only have to run before testing__. The `data_cleaning.py` is the file I used for data cleaning before the actual coding. The `testing.r` is the file I used for testing my code for research question 1 and 2 and the `test.py` is the python file I used for exporting testing datasets as well as testing my code for research question 4. The code used for testing is very simple to understand, and in the next paragraphs I will also provide detailed instructions to help readers using RStudio for testing.
## To set up the project, you need to:   
#### Open the `Anaconda.navigator`, click environment, then click CSE 163:
- __Install plotly:__ Please search for `plotly`, click the green checkmark, and click accept to install it
- __Install nltk:__ Please search for `nltk`, click the green checkmark, and click accept to install it
- __Install xgboost:__ Please search for `xgboost`, select all versions of the xgboost for python (there should be 4), click the green checkmarks, and click accept to install them
#### Open the unpacked project folder in Visual Studio Code, in the `terminal`:
(Note: Please be sure to __open the project folder and run the code inside the folder in the Visual Studio Code__, because all files in the code use the relative path.)
- Please type the word "python" in the termianl to enter the python mode
- Please type: import nltk
- Then, please type: nltk.download('punkt')
- Please exit python mode by typing: exit()
#### Download and set up RStudio for testing:
- __Install R in your laptop:__ https://cran.rstudio.com/
- __Download R-studio in your laptop:__ https://www.rstudio.com/products/rstudio/download/#download
#### Please open the RStudio, in the console, please type:
- install.packages('tidyverse')
- install.packages('geojsonio')
- install.packages('compare')
## To run the project, you need to:
- This is a re-emphasis: Please __open the project folder and run the code inside the folder in the Visual Studio Code__. You can do this by clicking *File* --> *Open Folder*, then the actual unpacked folder in Visual Studio Code.
- Open `final.py` in Visual Studio Code and run it for generating plots and machine learning models. (Note: This file may take some time to run, as building machine learning models is inherently a time-consuming process. On my computer, it takes about three minutes to run through the whole file.)
- Open `test.py` to export datasets we need to test to the folder and test the code for research question 4.
- Open `testing.r` in RStudio and click __source__ to run the whole file. (Note: Please don't click the run botton in RStudio because the run button will only run on the single line of code where the mouse is located. Instead, please click __source__ to run the entire code file.)
## Notice when running the code:
- There will be 5 web pages that pop up during the running of the code. This is because the plots generated using plotly need to be manually stored on the laptops. To save the plot generated from plotly, you can click the camera label on the top right of the popping website. Here, I have put all the saved and renamed generated plots into the folder, so there is no need for you to save them manually.
- There are three warnings about the version of the libraries I used in my code. I've talked to Wen and my TAs and finally we decided to ignore them. More detailed explanations of these three warnings are in my code with "#". When running the program, you will still see one FutureWarning. This is caused by importing xgboost. Since I need to comply with flake8, I cannot ignore the warning before importing the xgboost library. Therefore, this warning is still displayed at runtime. I have verfied this issue with Wen.
- In `final.py`, there are two places (in line 684 and line 729) where code for the print function is commented. These two places are the process of manually checking and filtering data in the code. The reason for keeping them off is to keep the terminal from looking messy when running the code. More detailed explanations of these two code lines are in my code with "#". I have verfied this action with Karen.
