from split_data import *
from songdata_train import *
from songdata_test import *


count = 0
add = 0
for i in range(0,100):
    splitme()
    traindata()
    accuracy = testdata()
    print(i)
    count += 1
    add += accuracy

print(count)
# print(add)
print(add/count)
