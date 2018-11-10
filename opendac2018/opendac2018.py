import json


assignment_train = './data/assignment_train.json'
pubs_train = './data/pubs_train.json'


##
# train: 'author_name' -> [ paper1, paper2, ... ]
# cluster: 'author_name' -> [ [id1, id2,...], [id1, id2,...], ... ]
# there are 100 authors needed to be disambiguated, ~24W appearing names, 147676 papers
# each author in training set has 318.22 clusters in average.

if __name__=="__main__":
    cluster = json.load(open(assigment_train,'r'))
    train_paper = json.load(open(ppubs_train,'r'))


