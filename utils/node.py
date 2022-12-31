
class Node:
    """
    attributes:
        feature [str]: column name in the dataframe
        split_kind [str]: '<=' or '>' if feature is numerical type,
                          '==' or '!=' if feature is categorical
        threshold [str, float]: if str, category being chosen,
                                if float value of the criteria used to split data
        info_gain [float]: information gain due to the split
        depth [int]: depth level of the node in the tree
        is_leaf [bool]: True if the node is a leaf of the tree
        left [Node]: node child where criteria is True
        right [Node]: node child where criteria is False
    """
    def __init__(self,
        feature=None,
        split_kind=None,
        threshold=None,
        info_gain=None,
        depth=0,
        is_leaf=False,
        left=None,
        right=None):

        # split_info
        self.feature = feature
        self.split_kind = split_kind
        self.threshold = threshold
        self.info_gain = info_gain

        self.is_leaf = is_leaf
        if self.is_leaf:
            self.content = f"[Leaf = feat:{self.feature} - kind: {self.split_kind} - threshold:{self.threshold} - info_gain: {self.info_gain}]"
        else:
            self.content = f"[Node = feat:{self.feature} - kind: {self.split_kind} - threshold:{self.threshold} - info_gain: {self.info_gain}]"
        # children nodes
        self.left_child = left
        self.right_child = right

        # meta
        self.depth = depth

    def __str__(self):
        if self.is_leaf:
            output_print = f"{self.content} --- node depth = {self.depth}"
        else:
            output_print = f"{self.content} --- leaf depth = {self.depth}"
        return output_print