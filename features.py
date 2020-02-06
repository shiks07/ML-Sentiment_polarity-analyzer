import sys

#unpacking the arguments
train_input,validation_input,test_input,dict_input,formatted_train_out,formatted_validation_out,formatted_test_out,feature_flag = sys.argv[1:]

# reading the data from a file 

def read_file(filename):
    f = open(filename,'r')
    data = [[y,x] for y,x in (line.strip().split("\t") for line in f)]
    return(data)

#model 1
def model1(data):
    features = [x.strip().split(" ") for y,x in data]
    features_sparse = []
    for j in range(len(features)):
        bag_of_words = {}
        bag_of_words[-1] = 1
        for i in range(len(features[j])):
            if features[j][i] in vocab.keys():
                index = vocab[features[j][i]]
                bag_of_words[index] = 1
        features_sparse.append(bag_of_words)
    return(features_sparse)

#model 2
def model2(data,t):
    features = [x.strip().split(" ") for y,x in data]
    features_sparse = []
    for j in range(len(features)): #j corresponds to jth example
        bag_of_words,bag_of_words_trim = {},{}
        bag_of_words[-1],bag_of_words_trim[-1] = 1,1
        for i in range(len(features[j])):
            if features[j][i] in vocab.keys():
                index = vocab[features[j][i]]
                if index in bag_of_words.keys():
                    bag_of_words[index] += 1
                else:
                    bag_of_words[index] = 1
        for i,v in bag_of_words.items():
            if v < t:
                bag_of_words_trim[i] = 1
        features_sparse.append(bag_of_words_trim)
    return(features_sparse)
    
#training data
data_train = read_file(train_input)
labels_train = [int(y) for y,x in data_train] 

#validation data
data_validation = read_file(validation_input)
labels_validation = [int(y) for y,x in data_validation]

#test data
data_test = read_file(test_input)
labels_test = [int(y) for y,x in data_test]

# dictionary
f = open(dict_input,'r')
vocab = tuple((x,int(y)) for x,y in (line.strip().split(" ") for line in f))
vocab = dict(vocab)
f.close()


if feature_flag == '1':
    features_train = model1(data_train)
    #data_train = list(zip(labels_train,features_train))
    features_test = model1(data_test)
    #data_test = list(zip(labels_test,features_test))
    features_valid = model1(data_validation)
    #data_valid = list(zip(labels_validation,features_valid))
else:
    features_train = model2(data_train,4)
    #data_train = list(zip(labels_train,features_train))
    features_test = model2(data_test,4)
    #data_test = list(zip(labels_test,features_test))
    features_valid = model2(data_validation,4)
    #data_valid = list(zip(labels_validation,features_valid))    
                      
# ouput functions

def formatted_output(features,labels):
    formatted_output = []
    for j in range(len(features)):
        temp = [str(labels[j])]
        for i,v in features[j].items():
            if i != -1:
                temp.append('\t'+str(i)+':'+str(v))
        formatted_output.append(temp)
    return(formatted_output)

def write_output(formatted_output,filename):
    fout = open(filename,'w')
    for i in formatted_output:
        fout.writelines(i)
        fout.write('\n')
    fout.close()

# writing outputs
train_formatted = formatted_output(features_train, labels_train)
write_output(train_formatted,formatted_train_out)

test_formatted = formatted_output(features_test, labels_test)
write_output(test_formatted,formatted_test_out)

valid_formatted = formatted_output(features_valid, labels_validation)
write_output(valid_formatted,formatted_validation_out)






