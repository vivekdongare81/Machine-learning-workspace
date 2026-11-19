import numpy as np
print(np.__version__)

print("================ 1 - basics ====================")
my_list = [1,2,3,4,5]
my_list = my_list*2
print(my_list)

my_np_list = np.array([1,2,3,4,5])
my_np_list = my_np_list * 2 
print(my_np_list)
print(type(my_np_list))

print("================ 2 - access ====================")

array = np.array('A')
print(array.ndim)
array = np.array(['A','B','C'])
print(array.ndim)
array = np.array([['A','B','C'], 
                  ['A','B','C'],
                  ['A','B','C']])
print(array.ndim)
array = np.array([[['A','B','C'], ['A','B','C'],['A','B','C']],
                  [['A','B','C'], ['A','B','C'],['A','B','C']]])
print(array.ndim)
print(array.shape)# row col data
print(array[0][0][0]) # chain indexing
print(array[0,0,0]) # fast - multidim indexing
print(array[0,0])

print("================ 3 - slicing ====================")
array = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]])
#array[start : end : step(optional)]
print(array[0:3])
print(array[0:3:2])
print(array[:])
print(array[::-1])
print(array[::-2])

print(array[0,0])
print(array[:,0])
print(array[:,:])
print(array[:, 0:3])
print(array[0:2, 0:2])

print("================ 4 - arithmetic ====================")

#Scalar Arithmetic
array = np.array([1,2,3])
print(array)
print(array+1)
print(array-1)
print(array*2)
print(array/2)
print(array**2)

#Vectorized Math Functions
radius = np.array([1,2,3])
print(np.sqrt(radius))
print(np.round(radius))
print(np.pi)

print(np.pi * radius **2)# apply Area formula on all nums in array A = pi*r^2

# Element-wise arithmetic
array1 = np.array([1,2,3])
array2 = np.array([4,5,6])
print(array1 + array2)
print(array1 - array2)
print(array1 * array2)
print(array1 / array2)
print(array1 ** array2)

# Comparision Operators
score = np.array([91,55,100,73,82,64])
score[score < 60] = 0 # filtering
print(score)

print("================ 5 - broadcasting ====================")

#Defination: Broadcasting is a powerful mechanism that allows NumPy to work with arrays of different shapes when performing arithmetic operations.
# Rules of Broadcasting:
# 1. If the arrays do not have the same rank, prepend the shape of the smaller rank array with 1s until both shapes have the same length.
# 2. The two arrays are said to be compatible in a dimension if they have the same size in that dimension, or if one of the arrays has size 1 in that dimension.

array1 = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
array2 = np.array([1,2,3])
print(array1.shape)
print(array2.shape)

array3 = array1 + array2# array2 is broadcasted to match the shape of array
print(array3)
print(array3.shape)

array1 = np.array([[1,2,3,4]])
array2 = np.array([[1],[2],[3],[4]])
print(array1.shape)#shape gives the dimensions of the array like (4,) means 1D array with 4 elements, (4,1) means 2D array with 4 rows and 1 column
print(array2.shape)
array3 = array1 + array2# array1 is broadcasted to match the shape of array2
print(array3)  
print(array3.shape)

print(array1 * array2)

print("================ 6 - aggregate function ====================")

#defination: Aggregate functions are functions that operate on a set of values and return a single value. In NumPy, aggregate functions are used to perform operations on arrays, such as finding the sum, mean, or maximum value of the elements in the array.
array = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
print(array.sum())
print(array.sum(axis=0))# sum of each column
print(array.sum(axis=1))# sum of each row
print(array.mean())
print(array.max())
print(array.min())
print(array.std())# standard deviation
print(array.var())# variance
print(array.cumsum())# cumulative sum
print(array.cumprod())# cumulative product
print(array.argmin())# index of min value
print(array.argmax())# index of max value
print(array.argsort())# index of sorted array
print(array.flatten())# flatten the array to 1D
print(array.ravel())# flatten the array to 1D
print(array.T)# transpose the array
print(array.reshape(1,9))# reshape the array to 1D
print(array.reshape(9,1))# reshape the array to 2D
print(array.reshape(3,3))# reshape the array to 2D
print(array.reshape(3,3,1))# reshape the array to 3D

print("================ 7 - filtering ====================")
#defination: Filtering is the process of selecting a subset of data from an array based on certain conditions. In NumPy, filtering can be done using boolean indexing, where a boolean array is used to select elements from the original array.
array = np.array([1,2,3,4,5,6,7,8,9,10])
print(array[array > 5])# filter elements greater than 5
print(array[array < 5])# filter elements less than 5
print(array[array % 2 == 0])# filter even elements
print(array[array % 2 != 0])# filter odd elements
print(array[(array > 5) & (array < 8)])# filter elements greater than 5 and less than 8
print(array[(array < 5) | (array > 8)])# filter elements less than 5 or greater than 8

array = array[array > 5]# filter elements greater than 5 and assign to array
print(array)

#Where function: The where function is used to return the indices of elements in an array that satisfy a given condition. It can also be used to replace elements in an array based on a condition.
array = np.array([1,2,3,4,5,6,7,8,9,10])
print(np.where(array > 5))# return the indices of elements greater than 5
print(np.where(array > 5, array, 0))# replace elements greater than 5 with their values and others with 0

print("================ 5 - Random Numbers ====================")

rng = np.random.default_rng() # create a random number generator
print(rng.random(5)) # generate 5 random numbers between 0 and 1
print(rng.integers(low = 1, high = 10, size=5)) # generate 5 random integers between 1 and 10
print(rng.integers(1, 10, size=(2, 3))) # generate a 2x3 array of random integers between 1 and 10
print(rng.uniform(1, 10, size=5)) # generate 5 random numbers from a uniform distribution between 1 and 10
print(rng.normal(low = 0, high = 1, size=5)) # generate 5 random numbers from a normal distribution with mean 0 and standard deviation 1
print(rng.choice([1, 2, 3, 4, 5], size=5, replace=True)) # generate 5 random numbers from a given list with replacement
print(rng.choice([1, 2, 3, 4, 5], size=5, replace=False)) # generate 5 random numbers from a given list without replacement     
print(rng.shuffle([1, 2, 3, 4, 5])) # shuffle a given list
print(rng.permutation([1, 2, 3, 4, 5])) # return a new array with the elements of the given list shuffled
print(rng.permutation(5)) # return a new array with the elements of the given range shuffled
print(rng.permutation(5).reshape(5,1)) # return a new array with the elements of the given range shuffled and reshaped to 5x1


