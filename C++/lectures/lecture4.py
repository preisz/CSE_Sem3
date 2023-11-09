#phython: ref counting for cleanup
#counts handles
#if mot referenced then can be deleted

lst =[1,2,3,4] #ref count of "lst" is 1
lst2=[1,2,3] #ref count of "lst2" is 1

lst = lst2 #ref count if "lst" resources goes to 0
 

def func(lst):
    pass
