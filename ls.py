import numpy as np

class Node(object):
    def __init__(self, parent, data_index, predict_value):
        self.parent = parent
        self.data_index = data_index
        self.predict_value = predict_value
        self.left = None
        self.right = None
        self.split_feature_index = None
        self.split_feature_value = None
        
class RegressionTree1(object):
    def __init__(self, min_data_in_leaf):
        self.min_data_in_leaf = min_data_in_leaf
        self.tree = None
            
    def fit(self, x, y):
        # 创建根节点
        candidate_node = []
        tree = Node(None, np.arange(x.shape[0]), np.mean(y))
        candidate_node.append(tree)
        
        # 遍历构建回归树，广度遍历
        while candidate_node:
            temp = candidate_node.pop(0)
            print('当前节点的索引：', temp.data_index)
            if len(temp.data_index) <= self.min_data_in_leaf:
                continue
            t = x[temp.data_index]
            t_y = y[temp.data_index]
            
            min_index_split = np.inf
                    
            # 遍历各个特征，找到使得数据平方差之和最小的特征，并从平均值划分左右节点
            for i in range(t.shape[1]):
                argsort = np.argsort(t[:, i])
                for split in range(1, len(argsort)):  # 左右两端不需要考虑
                    c1 = t[argsort[:split]]
                    c2 = t[argsort[split:]]
                    m1 = np.mean(t_y[argsort[:split]])
                    m2 = np.mean(t_y[argsort[split:]])
                    temp_loss = np.sum(np.power(y[argsort[:split]]-m1, 2)) + np.sum(np.power(y[argsort[split:]]-m2, 2))
                    if temp_loss < min_index_split:
                        min_index_split = temp_loss
                        temp.split_feature_index = i
                        temp.split_feature_value = x[:, i][split]
                        temp_left_index = argsort[:split]
                        temp_right_index = argsort[split:]
                        
            print('左右节点的索引：', temp_left_index, temp_right_index)
                        
            # 获取左右节点的数据索引
            left_index = temp.data_index[temp_left_index]
            right_index = temp.data_index[temp_right_index]
            
            print('左右节点的真实索引：', left_index, right_index)
            
            # 构造左右两个节点，并入队列
            temp.left = Node(temp, left_index, np.mean(y[left_index]))
            temp.right = Node(temp, right_index, np.mean(y[right_index]))
                                  
            candidate_node.append(temp.left)
            candidate_node.append(temp.right)
                                  
        # 返回根节点
        self.tree = tree

    def predict(self, x):
        y = []
        for x_i in x:
            node = self.tree
            while node.left is not None and node.right is not None:
                if x_i[node.split_feature_index] <= node.split_feature_value:
                    node = node.left
                else: node = node.right
            y.append(node.predict_value)
        return np.array(y)

      