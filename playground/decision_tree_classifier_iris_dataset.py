from utils.decision_tree_classifier import MyDecisionTreeClassifier, MyDecisionTreeRegressor
from utils.node import Node
from sklearn.datasets import load_iris
import pandas as pd

def ft_print_node(node):
    if isinstance(node.left_child, Node):
        ft_print_node(node.left_child)
    if isinstance(node.right_child, Node):
        ft_print_node(node.right_child)
    
    print(node)

if __name__ == '__main__':

    data_iris = load_iris()
    X = pd.DataFrame(data=data_iris.data, columns=data_iris.feature_names)
    y = pd.DataFrame(data=data_iris.target, columns=["target"])
    df = pd.concat([X, y], axis=1)
    for i, label in enumerate(data_iris.target_names):
        df.loc[df['target'] == i, 'target'] = label

    my_classifier = MyDecisionTreeClassifier(max_depth=3, min_samples_split=5)
    my_classifier.fit(data=df, target="target")
    print('type de my_classifier.tree: ', type(my_classifier.tree))
    ft_print_node(my_classifier.tree)