import pickle

# open a file, where you stored the pickled data
file = open('C:/Users/tyler/europa/k2Q_N2vec_L80_N80g6_tau9_mpi_H300.pickle', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print('Showing the pickled data:')

cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    cnt += 1