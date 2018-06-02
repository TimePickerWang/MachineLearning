import MachineLearning.DecisionTree.trees as tree
import pandas as pd


def creat_test_tree():
    data = pd.read_csv("./Data/lenses.txt", sep='\t', header=None)
    lenses = data.as_matrix().tolist()
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = tree.creat_tree(lenses, lenses_labels)
    # tree.store_tree(lenses_tree, "./Data/underGlassTree.txt")  # 存储树


# creat_test_tree()
my_tree = tree.grab_tree("./Data/underGlassTree.txt")
lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
test_vect1 = ['presbyopic', 'hyper', 'no', 'normal']
test_vect2 = ['young', 'hyper', 'yes', 'normal']
print(tree.classify(my_tree, lenses_labels, test_vect1))
print(tree.classify(my_tree, lenses_labels, test_vect2))
