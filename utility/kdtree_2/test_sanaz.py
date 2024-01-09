import kdtree as kd
import numpy as np
import ast

def test(points):
    tree = kd.create(points)
    result = tree.search_knn([0.5,0.5],1)
    #result=result[0].replace(' ','')
    tmp = result[0][0].data

    #out = ast.literal_eval(tmp)
    print(f"out {tmp}")
    #tree.add([3,3])
    #tree.add([0.5,0.5])

    """n = 2000000
    new_points = np.random.uniform(0.5, 1.5, (n, 6))
    for i in range(n):
        tree.add(new_points[i])

    print(f"Finished adding {n} nodes")

    for i in range(n):
        result = tree.search_knn(new_points[i],1)
        print(f"node {new_points[i]} ---> {result}")"""

    #kd.visualize(tree)




def main():
    points = [[0,0], [0,1], [0,2], [1,2], [1,1], [2,2]]
    #points = [[0,0,0,0,0,0]]
    test(points)



main()

