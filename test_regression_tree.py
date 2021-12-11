from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from sklearn.datasets import make_regression
from regression_tree import RegressionTree
from ls import RegressionTree1


def draw_tree(regression_tree, filename):
    # graphvizで木構造を出力する
    G = Digraph(format="png")
    que = Queue()
    que.put((regression_tree.tree, 0))
    while not que.empty():
        top, node_num = que.get()
        # leaf node
        if top.split_feature_index is None:
            G.node(str(node_num), "{0:.3f}".format(top.predict_value), shape="doublecircle")
        # internal node
        else:
            G.node(str(node_num), "{0}:{1:.3f}".format(top.split_feature_index, top.split_feature_value), shape="circle")
            if top.left is not None:
                G.edge(str(node_num), str(node_num * 2 + 1), label="<")
                que.put((top.left, node_num * 2 + 1))
            if top.right is not None:
                G.edge(str(node_num), str(node_num * 2 + 2), label=">=")
                que.put((top.right, node_num * 2 + 2))
    G.render(filename)


if __name__ == '__main__':
#     rt_1 = RegressionTree1(1)
#     rt_3 = RegressionTree1(3)
    rt_5 = RegressionTree1(5)
    N = 20
    D = 1
    X, ys = make_regression(n_samples=N, n_features=D, random_state=0, noise=4.0, bias=100.0)
    print(X, ys)
#     rt_1.fit(X, ys)
#     rt_3.fit(X, ys)
    rt_5.fit(X, ys)
#     draw_tree(rt_1, "rt_1")
#     draw_tree(rt_3, "rt_3")
    draw_tree(rt_5, "rt_5")

    X_test = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
#     ys_predict_1 = rt_1.predict(X_test)
#     ys_predict_3 = rt_3.predict(X_test)
    ys_predict_5 = rt_5.predict(X)
#     plt.scatter(X, ys, color="black", label="true")
#     plt.plot(X_test, ys_predict_1, color="red", label="min_data_in_leaf=1")
#     plt.plot(X_test, ys_predict_3, color="green", label="min_data_in_leaf=3")
#     plt.plot(X_test, ys_predict_5, color="blue", label="min_data_in_leaf=5")
#     plt.legend()
#     plt.savefig("fig_test_regression_tree.png")
    print('均方误差',np.sum(np.power(np.array(ys)-np.array(ys_predict_5), 2))/20)
    print('均绝对误差',np.sum(np.absolute(np.array(ys)-np.array(ys_predict_5)))/20)
    a = np.absolute(np.array(ys)-np.array(ys_predict_5))
    print('绝对比列误差',np.sum(a/np.array(ys))/20)
    print("complete")
