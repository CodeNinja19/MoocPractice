import random as rn
import math
import numpy as np

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

# find(), index(), isDigit(), isalpha(), lower(), upper(), islower(), isupper(), strip(), replace
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

## MISCLLENOUS FUNCTION AND OTHER STUFF ##

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

# np.sum(a) for array sum

# np.mean(a) for array mean

# np.max(a) and np.min(a) for max and min of the array





###########################################################################################################################


## MAGIC SQUARE ##

# sum all the elements in any row,col and diagonal is same.

# magic constant = sum of a row or col or diag.  (sumOfAllElements/no.rows)

# ramanujan magic constant 139.

# ramanujan multiplicative magic constant 1729.


##########################################################################################################################

## AUDIO ##

# .wav or .wave files contain raw/high quality audio data. They are uncompressed files
# librarys that can be used to work with them are wave, scipy.io.wavefile, pydub

## SUBSTITUTION CIPHER ##

# a cIpher in which each letter is replace with an specific letter mostly based on a key.
# CEASAR CIPHER is a special example of substitution cipher where each alphabet is shifted by a specific 
# value giving us an cipher.

## Frequency analysis of substitution cipher ##

# Here in natural language some character are more frequenty then some.
# So we replace the most frequent ciphertext letter most likely plaintext counter-part.
# we also use common letter pairing to refine accuracy.

# It requires a large text for accuracy.
# If the text encrypted using any other cypher then the common pattern in language can disappear leading to 
# drawbacks.



###########################################################################################################################

