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

# Random Forest Algorithm
#responsible for creating the samples of the training dataset, training a decision tree on each,
#then making predictions on the test dataset using the list of bagged trees.
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):

def get_split(dataset, n_features):
    ##Given a dataset, we must check every value on each attribute as a candidate split,
    #evaluate the cost of the split and find the best possible split we could make.

            ##When selecting the best split and using it as a new node for the tree
            #we will store the index of the chosen attribute, the value of that attribute
            #by which to split and the two groups of data split by the chosen split point.
            ##Each group of data is its own small dataset of just those rows assigned to the
            #left or right group by the splitting process. You can imagine how we might split
            #each group again, recursively as we build out our decision tree.

    ##Once the best split is found, we can use it as a node in our decision tree.
    ##We will use a dictionary to represent a node in the decision tree as
    #we can store data by name.


def split(node, max_depth, min_size, n_features, depth):
    #Firstly, the two groups of data split by the node are extracted for use and
    #deleted from the node. As we work on these groups the node no longer
    #requires access to these data.

    #Next, we check if either left or right group of rows is empty and if so we create
    #a terminal node using what records we do have.
    # check for a no split

    #We then check if we have reached our maximum depth and if so we create a terminal node.
    # check for max depth

    #We then process the left child, creating a terminal node if the group of rows is too small,
    #otherwise creating and adding the left node in a depth first fashion until the bottom of
    #the tree is reached on this branch.
    # process left child

    # process right child
    #The right side is then processed in the same manner,
    #as we rise back up the constructed tree to the root.
