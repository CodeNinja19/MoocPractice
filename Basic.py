import random as rn
import math
import numpy as np
import turtle
from PIL import Image
import datetime
import calendar

a="ab\rcd"
b="abcda"
print(a)
print(b.isdigit())
print(b.isalnum())
print(b.isalpha())
print(b.upper())
print("aaaa".count("a"))
print("aaba".find("ba"))
print("aaba".replace("b","c"))

# String functions

# Slicing, string are immutatble and f strings 

# methods 

# find(), index(), isDigit(), isalpha(), lower(), upper(), islower(), isupper(), strip(), replace()
# lstrip(), rstrip(), capitalize(), startswith(), endswith()

# index== gives error is substring not found

###################################################################################################
# representing octal number
# num=0O123

print(int(oct(146),8))


for i in range(10):
    print(i,end=' ')

###################################################################################################

#List#

a=[1,2,2,3,4,5,6,6]

print(a)
a.append(7)
print(a)

print(a.index(7))
print(sorted(a))
#methods#

# append(), index(), clear(), remove(), pop(), insert(), reverse(), sort(), count(), extend()
# clear()

#sort( reverse=[true,false])

# sorted and reversed are functions
# index will give error if the element is not found 

# member ship operator==(in)

# functions

# sum(), max(), min(), len(), any(), all()

''' all(): Returns True if every element is True.
any(): Returns True if at least one element is True '''

#zip(): combines lists
# enumerate(): index - value pair  we then do tuple unpacking on them

# List comprehension

sqrA=[i*i for i in a]
print(sqrA)


############################################################################################################


## TUPLES ##

# Just immutable lists with fewer methods 

# declared using ()

tup=(1,2,3)

print(tup)

# methods

# count(), index() 

#functions 

# max(), min(), sum(), all(), any(), sorted(), reversed()

###########################################################################################################


## SETS ##

## to make 
## immutable and unordered 

# set= set()  to make empty set

st={1,2,3}  # or st=set([1,2,3])

print(1 in st)

# methods count(), add(), clear(), 

# discard(): deleted without error
# remove(): error if no element found
# pop removes a random element from the set and error if set is empty


# special functions: intersection(), union(), issubset()



############################################################################################################

## DICTIONARIES ##

# unordered key value pairs

# by default {} is an empty dictionary

#d=dict() or {}

# methods

# get(), items(), keys(), values(), clear(), len()

# to delete a key
# del d[key]
# d.pop(key)

d={chr(i+65):i for i in range(26)}

print(d.items())


############################################################################################################


## FILE HANDLING ##

file = open("./mooc.txt","w+")

file.writelines([f"hello world{x}\n" for x in range(10)])

file.close()


with open("./mooc.txt","r+") as file:
    print(file.read())

    file.seek(0,2)   

    file.write("Read this file!!")


##############################################################################################################

## MISCELLANEOUS FUNCTION AND OTHER STUFF ##

# MAP AND FILTER #
# (callback,iterable)  --> will return iterable not a list 
def sqr(i):
    return i*i

print([int(i) for i in map(sqr,[i for i in range(1,11)])])

## CLASSES ##

class myfirstclass:
    def __init__(self,x):
        self.x=x

    def printName(self):
        print(self.x)

obj1=myfirstclass("Kartik")

obj1.printName()

print("Hello"+3*"world")

## order importance

# random library
# numpy
# Graph (Finding degree of separartion & Finding number of edges questions)
# scratch and turtle game (2-3 questions only)


# all the remaining stuff in the course
# all the things and concepts in the course
# csv module 
# gmplot purpose
# turtle module
# image opening using Pillow library
# selenium
# date and time module
# calender module
# matplotlib
# pandas

##########################################################################################################

## RANDOM LIBRARY ##

rn.seed(10)
rint=math.floor(rn.random()*100)+1
rint2=rn.randint(10,11)
samples=[int(((i-3)*(i-2))/(i)) for i in range(1,101,5)]
print(samples)
print(rint,rint2,sep="      ")
print(rn.uniform(10,11))
print(rn.randrange(0,31,3))
print(rn.choices(samples,weights=[i**2 for i in range(1,len(samples)+1)],k=4))
rn.shuffle(samples)


'''
random.random()	                    Random float in [0.0, 1.0)
random.uniform(a, b)	            Random float in [a, b]
random.randint(a, b)	            Random integer in [a, b]
random.randrange(start, stop, step)	Random integer from start to stop - 1
random.choice(sequence)	            Random element from a sequence
random.choices(sequence, k=n)	    n random elements (with replacement)
random.sample(sequence, k=n)	    n unique random elements (without replacement)
random.shuffle(sequence)	        Shuffle sequence in place
random.seed(a)	                    Set the seed for reproducibility
random.gauss(mu, sigma)	            Random float from Gaussian distribution
random.betavariate(alpha, beta)	    Random float from Beta distribution
'''


#############################################################################################################

## NUMPY ##

# numpy array are homogenous in data type

npd=np.array([1,2,3])

# npd1=np.full((100,100),99)

npd1=np.arange(1,10001).reshape(100,100)

'''

np.zeros(shape):                    Creates an array filled with zeros
np.ones(shape):                     Creates an array filled with ones
np.eye(N):                          Creates an identity matrix of size N
np.full(shape, fill_value):         Creates an array filled with a specific value

shape=(row,col)
'''

print(npd.dtype)
print(npd)

print(npd1)

npd1=npd1.reshape((200,50))

print(npd1)

print(npd1.shape)

print(np.arange(1,11,2).reshape(5,1)*np.arange(2,11,2).reshape(5,1)) # a*b for element-wise multiplication

print(npd1.flatten())

print(npd1[0,0], npd1[0][0],sep="\t")

print(npd1[2:5,2:5])

print(np.linspace(1,10,5))

# boolean masking

print(npd1[npd1>500])

# broadcasting

print(npd+100)
print(np.array(list(map(lambda x:x**4,npd))))


# linear algebra

## IMP ##
'''
[1,2,3] is 1 D array so its shape returns (3,) and hence will remain same on transpose

[[1,2,3]] is 2 D array so its shape returns (1,3)
'''
print(npd.shape)
print(npd.T)

npd2=np.array([[1,2,3]])
print(npd2.shape)
print(npd2.T)  # or np.transpose(npd2)

# np.linalg.det(a) for determinant
# np.linalg.inv(a) for inverse

'''

np.arange(start, stop, step):   Similar to range in Python, but returns an array
np.linspace(start, stop, num):  Returns an array of evenly spaced values between start and stop

'''

'''

np.save(filename, array):   Saves an array to a .npy file
np.load(filename):          Loads an array from a .npy file

'''

# np.flatten() to make it linear

# np.dot(a,b) for matrix multiplication of a and b

# np.sum(a,axis=) for array sum

# np.mean(a) for array mean

# np.max(a) and np.min(a) for max and min of the array


# axis=0 for each column
# axis=1 for each row

###########################################################################################################################


## MAGIC SQUARE ##

# sum all the elements in any row,col and diagonal is same.

# magic constant = sum of a row or col or diag.  (sumOfAllElements/no.rows)

# ramanujan magic constant 139. (22,12,18,87)

# ramanujan multiplicative magic constant 1729.


##########################################################################################################################

## AUDIO ##

# .wav or .wave files contain raw/high quality audio data. They are uncompressed files
# librarys that can be used to work with them are wave, scipy.io.wavefile, pydub

## SUBSTITUTION CIPHER ##

# a cipher in which each letter is replace with an specific letter mostly based on a key.
# CEASAR CIPHER is a special example of substitution cipher where each alphabet is shifted by a specific 
# value giving us an cipher.

## Frequency analysis of substitution cipher ##

# Here in natural language some character are more frequenty then some.
# So we replace the most frequent ciphertext letter most likely plaintext counter-part.
# we also use common letter pairing to refine accuracy.

# It requires a large text for accuracy.
# If the text encrypted using any other cipher then the common pattern in language can disappear leading to 
# drawbacks.

###########################################################################################################################


## CSV FILES ##

# can use csv module
# with open("file.csv",mode="r") as file:
#   read =csv.reader(file)
#   for row in read:
#       print(row)

# can use pandas too. pandas.read_csv("file.csv")  will read it into a dataframe

## gmplot ##

# it helps plotting data on google map using python.
# gmap=gmplot.gmplot(latitude=,longitude=,value)


## TURTLE MODULE ##

'''
# Create a screen
screen = turtle.Screen()
screen.bgcolor("lightblue")  # Optional background color

# Create a turtle
t = turtle.Turtle()
t.shape("turtle")  # Choose shape ('turtle', 'arrow', 'circle', etc.)
t.color("darkgreen")  # Color of the turtle
t.speed(2)  # Speed of drawing (1 - slow, 10 - fast, 0 - no animation)

'''

# Moving

'''
t.forward(100)  # Move forward by 100 pixels
t.backward(50)  # Move backward by 50 pixels
t.right(90)     # Turn right by 90 degrees
t.left(45)      # Turn left by 45 degrees

'''

# Pen control

'''
t.penup()       # Lift the pen (stop drawing)
t.pendown()     # Lower the pen (start drawing)
t.pensize(3)    # Set pen thickness
t.color("blue") # Set pen color

'''

# Changing shape 

'''
t.shape("circle")   # Change turtle shape
t.color("purple")   # Change turtle color

'''

# going to a position without drawing

'''
t.penup()
t.goto(-100, 100)  # Move turtle to (x, y) coordinates
t.pendown()

'''

# to end the game 

'''

# To end turtle.done()

or close the window on click

screen.exitonclick()
'''
'''
screen=turtle.Screen()
screen.bgcolor("blue")

t=turtle.Turtle()

t.color("black")
t.shape("turtle")
t.speed(2)
t.penup()
t.begin_fill()
for i in range(8):
    if (i&1):
        t.pendown()
    t.forward(100)
    t.right(45)
    t.penup()

t.end_fill()
t.pendown()
t.goto(0,100)
t.write("Hello world")
screen.exitonclick()
'''
# in turtle by default pen is down

## PILLOW ##

# for opening, croping, etc of images
'''
img=Image.open("peacockFeather.jpg")
img.show()
print(img.format)
print(img.size)
print(img.mode)

'''

# methods

# img.resize() and img.crop() are imp to me

###########################################################################################################################

# VADER is generally used in sentimental analysis to detect emotional intensity

# PIL (Pillow) is used in image enhancement.

# a significant information can be extracted from a image using image enhancemenet.

###################################################################################################################

## FLAMES ##

# write both the names
# remove common characters from the names
# removes letters from FLAMES in a cycle using the remaining count of letters

## SELENIUM ##

# to automate the web browser
# webdriver.chrome() is used to control the web browser programmatically.
# driver.get(url) opens a specific url in web browser
# input field.send keys(Keys.RETURN) : to simulate pressing the enter key

## DATE TIME AND CALENDER ##

# datetime.datetime.now() for current date and time
print(datetime.datetime.now())


# Current date
today = datetime.date.today()  # e.g., 2024-11-01
print(today)
# Specific date
specific_date = datetime.date(2023, 12, 25)  # Christmas 2023
print(specific_date)
# Current time
now = datetime.datetime.now()  # e.g., 2024-11-01 12:30:45.123456
print(now)
# Specific time
specific_time = datetime.time(14, 30)  # 2:30 PM
print(specific_time)
# print(calendar.month())

print(today + datetime.timedelta(days=456))


print(calendar.month(2024,11))  # calendar for the month of a particular year

print(calendar.weekday(2024,11,2))  # corresponding weekday for a date

print(calendar.day_name[calendar.weekday(2024,11,2)])  # day name of a weekday

print(calendar.calendar(2024))  # calendar for a specific year


##############################################################################################################################

'''
Collatz conjecture is an unsolved problem in mathematics. 

If n== even: n=n/2
else n=3*n+1

it is claimed that n will reach 1 eventually.

Checked till 5.76 x 10^18
'''

# google use random walk simulation to rank web pages.
# web pages are ranked on the basis of random walk.
# PageRank relies heavily Hyperlink network between web pages.
# degree of separation: distance btw nodes.

# to draw on screen:                                turtle
# audio:                                            wave, scipy, pydub
# sentimental analysis:                             VADER
# image display/enchancement:                       PIL (Pillow)
# plot data on google maps:                         gmplot
# csv:                                              csv, pandas (pd.read_csv())
# plots:                                            matplotlib.pyplot
# data:                                             numpy
# nlp:                                              nltk    nltk.download() for downloading necessary packages
# for working with graphs related to networks:      NetworkX
# for visualizing and analyzing large networks:     Gephi
# to generate random numbers:                       random
# to automate web browsers                          selenium