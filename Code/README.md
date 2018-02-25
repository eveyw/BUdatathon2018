################################################################################
## Calculate the Gini index for a split dataset
################################################################################
## this is the name of the cost function used to evaluate splits in the dataset.
# this is a measure of how often a randomly chosen element from the set
#would be incorrectly labeled if it was randomly labeled according to the distribution
#of labels in the subset. Can be computed by summing the probability
#of an item with label i being chosen times the probability
#of a mistake in categorizing that item.
#It reaches its minimum (zero) when all cases in the node
#fall into a single target category.
#A split in the dataset involves one input attribute and one value for that attribute.
#It can be used to divide training patterns into two groups of rows.
#A Gini score gives an idea of how good a split is by how mixed the classes
#are in the two groups created by the split.
#A perfect separation results in
#a Gini score of 0,
#whereas the worst case split that results in 50/50 classes
#in each group results in a Gini score of 1.0 (for a 2 class problem).
#We first need to calculate the proportion of classes in each group.
def gini_index(groups, class_values):

# Split a dataset into k folds
# the original sample is randomly partitioned into k equal sized subsamples.
#Of the k subsamples, a single subsample is retained as the validation data
#for testing the model, and the remaining k − 1 subsamples are used as training data.
#The cross-validation process is then repeated k times (the folds),
#with each of the k subsamples used exactly once as the validation data.
def cross_validation_split(dataset, n_folds):

#Create child splits for a node or make terminal
#Building a decision tree involves calling the above developed get_split() function over
#and over again on the groups created for each node.
#New nodes added to an existing node are called child nodes.
#A node may have zero children (a terminal node), one child (one side makes a prediction directly)
#or two child nodes. We will refer to the child nodes as left and right in the dictionary representation
#of a given node.
#Once a node is created, we can create child nodes recursively on each group of data from
#the split by calling the same function again.
def split(node, max_depth, min_size, n_features, depth):

#Build a decision tree
######
# Building the tree involves creating the root node and
# calling the split() function that then calls itself recursively
# to build out the whole tree.
def build_tree(train, max_depth, min_size, n_features):

# Select the best split point for a dataset
# This is an exhaustive and greedy algorithm
def get_split(dataset, n_features):

a = set(a)
a.split(' ')


for index, row in dataFrame.iterrows():
    for row_item in row:
        print(row_item)
