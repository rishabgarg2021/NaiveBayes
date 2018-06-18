import pandas as pd
import random
import copy
import itertools
import numpy as np
from collections import defaultdict


from  collections import  defaultdict as dd

#preprocess of data takes place with removal of unknown data or entries filled by "?"
def preProcess():

    df=pd.read_csv("mushroom.csv",header=None)
    return df


#training of supervised data is done by creating the dict of dict of dict of probabilities of
#each columns class different attributes count. Column1:{ClassA:{Inst:0.4 ,Inst1: 0.6}}}
def train_supervised(df):

    #collects all the headers of columns which are just numbered from 0..n
    headers=list(df.columns.values)

    #exclude the last column as its the class attribute
    headers=headers[:len(headers)-1]

    #created dict for storing the columns initilised with 0,1,...n
    dict={}

    #gather all the unique values from dataFrame to help making dictionary
    classes=df[df.columns[len(df.columns)-1]].unique()

    #different unique classes possibilities is stored in classes list.
    classes=list(classes)

    #firslt initilize the dictionary with attributes.
    #e.g.: {attribute1:{class1:{},class2:{}...}  ,  attribute2:{class1:{}, class2:{}...}...}
    for h in headers:
        dict[h]={}
        for cl in classes:
            dict[h][cl]={}

    #each row is being traversed and frequency is written with number of occurences of each
    #instance in {attribute:{class:{instances:counts}}}
    for (index,row) in df.iterrows():
        # converts all the data into string to easily make keys of same dataType
        for i in range(len(row)-1):
            row[i]=str(row[i])

        # each row attribute is added to the dictionary with its number of occurunces.
        #if the value doesn't exist the dict is created with count 1.
        j=-1
        for h in headers:
            j+=1
            try:

                dict[h][row[len(row)-1]][row[j]]+=1

            except KeyError:
                if (row[j] != "?"):
                    dict[h][row[len(row) - 1]][row[j]]=1



    #ADD 1 Soothing to help probability of given attributes instance to not exist of particular class
    #while calculating the rhe proability of each class of given instance.
    ## need to add 1 value to each of the given result and 1 if it doesn't exist to help probabilistic smoothing
    #it helps to avoid the case in which total result in naive bayes becomes 0 is an instance doesn't exist for
    #attribute with given class.

    for i in range(len(row)-1):

        un=df[df.columns[i]].unique()

        newUN=[]
        for j in range(len(un)):
            if(un[j]!="?"):
                newUN.append(un[j])
        un=[]
        un=newUN



        for clas in classes:
            for inc in un :
                try:

                    dict[i][clas][inc]+=1

                except KeyError:

                    dict[i][clas][inc]=1


    #it assigns the probabilty of each  attribute with given classes the probability of all instances of attribute
    #e.g. row0 has class A with inst1=40, inst2=30,inst3=60.
    # so the row0:{classA:{inst1:(40/(40+30+60)),inst2:{30/(40+30+60)}} is created for testing data
    classCount=df[df.columns[-1]].value_counts()
    classDict=classCount.to_dict()
    for classes in dict.keys():

        uniqueClass = df[df.columns[classes]].unique()
        # print(uniqueClass)
        newUN = []
        for j in range(len(uniqueClass)):
            # print(j)
            if (uniqueClass[j] != "?"):
                newUN.append(uniqueClass[j])
        un = []
        uniqueClass = newUN

        for value in dict[classes].keys():
            for instance in dict[classes][value].keys():

                dict[classes][value][instance] = dict[classes][value][instance]\
                                                 / (classDict[value]+len(uniqueClass))

    #prediction of all the training data is again made
    return (dict,df,classDict)



#data is predicted for supervised learning by calculating the conditional probability
#arg max P(x1, x2, ..., xn|cj)P(c) and further checking the arg max value with the actual class.
def predict_supervised(dict,df,classDict):
    classes = df[df.columns[len(df.columns) - 1]].unique()
    classes = list(classes)
    correctNumber=0
    totalLength=len(df[df.columns[-1]])

    #class Dict helps to create the probability of different classes in class column of dataset.
    #prior proabilities are calculates here.
    for key in classDict.keys():
        classDict[key]=classDict[key]/totalLength

    #posterior proabilities are calculated and the highest class proability wins here.
    for (index, row) in df.iterrows():
        lis=[]
        for cl in classes:
            total=1
            for i in range(len(row)-1):
                if(row[i]!="?"):
                    try:
                        total = total * dict[i][cl][row[i]]
                    except KeyError:
                        pass
            lis.append((total*classDict[cl]))
        ind=lis.index(max(lis))
        predictClass=classes[ind]
        if(predictClass==row[len(row)-1]):
            correctNumber+=1

    #it prints out the accuracy for the supervised learning of data.
    print("accuracy of supervised learning :",(correctNumber/totalLength)*100)
    return (correctNumber/totalLength)*100


#unsuper data is trained by assigning proabilities of different class possibilities by choosing random number
#from 0-1 such that for a given instance proability of different classes sum to 1.
def train_unsupervised(df):

    first=True
    oldDf=df.copy()
    headers = list(df.columns.values)
    headers = headers[:len(headers) - 1]
    classes = df[df.columns[len(df.columns) - 1]].unique()
    classes = list(classes)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    newDf = df.copy()
    randomList=[]
    classRandom=[]

    #it heps to assign proability of instance a random number between o to 1 and then further assigning
    #such that the sum equals to 1 of a given row with different classes.
    for (index,row) in df.iterrows():
        min=0
        max=1
        total=0
        for i in range(len(classes)-1):
            n=random.uniform(min,max)
            total+=n
            max=1-total
            classRandom.append(n)
        classRandom.append((1-total))
        randomList.append(classRandom)
        classRandom=[]

    #it helps to create the different combinations of given class such that checking each combination
    #the combination of class which gives the best accuracy for the first time is treated as the best
    #for assigning the random proabilities of a given class and that best order is maintained throughout
    #the iterations to further improve the proability.
    differentClass=differentClassOrder(classes)
    accuracy=[]
    for classOrder in differentClass:

        #it helps to give the accuracy with different class orders.
        #first helps to create the proabilities created in random list to dataframe to further manipulate
        #in predicting the accuracy for the first time.

        (acc, randomList1)=trainUtilData(classOrder,randomList,df,headers,newDf,oldDf,first)
        first=False
        accuracy.append(acc)



    #it helps to get the best accuracy index of different combinations tried with different class order.
    #for same random probabilities being created.
    bestClassIndex=np.argmax(accuracy)


    #helps to get the new class order with the matching index of best accuracy tested.
    newClassOrder=differentClass[bestClassIndex]

    #the random list is sent to trainUtilIteration which gives the newlist whcich is the new proabilities of
    #given class by consedering the proabilities sent randomly first time.
    (acc, newlist)=trainUtilDataIteration(newClassOrder,randomList,df,headers,newDf,oldDf,first)
    print("accuracy for unsupervised learning : ", acc)
    #10 different iterations are sent to calculate the accuracy of each instance with new prior probabilities of class
    #consedring each row posterior proabilities calculated at first instance to get different class proaboility.

    for i in range(10):
        (acc, newlist) = trainUtilDataIteration(newClassOrder, newlist, df, headers, newDf, oldDf,first)
        print("iteration",i," :",acc)



#it trains the unsupervise data by giving the accuracy of data by predicting.
def   trainUtilData(classes,randomList,df,headers,newDf,oldDf,first):
    classColumn = []
    column = 0
    classDict = {}

    #classDict helps to create the prior probabilities of classes.
    #first time it helps to create the dataframe with the unique class names as new attribute and then we need to change
    #the class order of the dataFrame according to newClassOrder.
    if(first):
        for cl in classes:
            for i in range(len(randomList)):
                classColumn.append(randomList[i][column])
            column += 1
            df[cl] = classColumn
            classDict[cl] = (df[cl].sum() / len(randomList))
            classColumn = []

    #dataframe class labels are changes with new class order.
    else:
        columnNames=list(df.columns.values)
        i=len(columnNames)-len(classes)
        col=columnNames[:i]
        for cl in classes:
            col.append(cl)
        df.columns=col

        for cl in classes:
            classDict[cl] = (df[cl].sum() / len(randomList))

   #helps to create the dictionary with the random proabilities total of same instances of given attribute
    #for a given class to help calculate the posterir probabilites.
    dict = {}

    for h in headers:
        dict[h] = {}
        for cl in classes:
            dict[h][cl] = {}
    for (index, row) in df.iterrows():
        # converts all the data into string to easily make keys of same dataType
        for i in range(len(row) - 1):
            row[i] = str(row[i])
        j = -1
        for h in headers:
            j += 1
            for cl in classes:

                try:

                    dict[h][cl][row[j]] += df.loc[index, cl]

                except KeyError:

                    dict[h][cl][row[j]] = df.loc[index, cl]

    (newlist,acc) = predict_unsupervised(dict, oldDf, classDict,classes)
    #returns the newlist with the accuracy calculated each time with new class order.
    return (acc,newlist)



#trains the unsupervised data with iterations consdering same proabilities are used as prior of which the
#randomlist proabilities were creared as posterior in predicting the data
def trainUtilDataIteration(classes,randomList,df,headers,newDf,oldDf,first):
    classColumn = []
    column = 0
    classDict = {}
    columnNames = list(df.columns.values)
    i = len(columnNames) - len(classes)
    col = columnNames[:i]

    #column name of dataFrame are again updated as the best combinations.
    for cl in classes:
        col.append(cl)
    df.columns = col

    #best class order is now treated to generate the classDict proabilities in usinf the prior proabilities.
    for cl in classes:
        for i in range(len(randomList)):
            classColumn.append(randomList[i][column])
        column += 1
        df[cl] = classColumn
        classDict[cl] = (df[cl].sum() / len(randomList))
        classColumn = []

    # helps to create the dictionary with the random proabilities total of same instances of given attribute
    # for a given class to help calculate the posterir probabilites.
    dict = {}

    for h in headers:
        dict[h] = {}
        for cl in classes:
            dict[h][cl] = {}

    for (index, row) in df.iterrows():

        for i in range(len(row) - 1):
            # converts all the data into string to easily make keys of same dataType
            row[i] = str(row[i])
        j = -1
        for h in headers:
            j += 1

            for cl in classes:

                try:

                    dict[h][cl][row[j]] += df.loc[index, cl]

                except KeyError:
                    if (row[j] != "?"):
                        dict[h][cl][row[j]] = df.loc[index, cl]

    (newlist,acc) = predict_unsupervised(dict, oldDf, classDict,classes)
    # returns the newlist with the accuracy calculated each time with new class order.
    return (acc,newlist)




#helps to get the permutations possible for the
def differentClassOrder(classes):

    classesOrder=[]
    classesOrder=list(itertools.permutations(classes))
    classesOrder = list(map(list,classesOrder))

    return classesOrder




#it predicts the data for UnsuperVised learning with dataFrame and same class order as the best or the different
#combination of classes order.
def predict_unsupervised(dict,df,classDict,classes):
    correctNumber = 0
    totalLength = len(df[df.columns[-1]])
    newList=[]
    #calculates the posterior proability for each class and choose the class with the maximum proability
    #and assigns the proabilities calculated for each class to newlist which can be used each time for
    #iteration also.

    for (index, row) in df.iterrows():
        lis = []
        for cl in classes:
            total = 1
            for i in range(len(row) - 1):
                if (row[i] != "?"):
                    total = total * (dict[i][cl][str(row[i])]/classDict[cl])
            lis.append((total * classDict[cl]))
        ind = lis.index(max(lis))
        newList.append([lis[x]/sum(lis) for x in range(len(lis))])

        #the best index with proabilities of different class is treated for predicting the class.
        predictClass = classes[ind]

        #if predictions are same as the actual class count is incremented.
        if (predictClass == row[len(row) - 1]):

            correctNumber += 1
    #returns the accuracy by checking the proportion of right instance with total instance.
    return (newList,(correctNumber / totalLength) * 100)
















#it helps to make dataFrame easily readable from csv format to dataFrame.
def evaluate_supervised():

    dataFrame=preProcess()
    dict, temp, classDict = train_supervised(dataFrame)
    acc = predict_supervised(dict, temp, classDict)
    print(acc)

def evaluate_unsupervised():
    dataFrame = preProcess()
    train_unsupervised(dataFrame)


evaluate_unsupervised()


