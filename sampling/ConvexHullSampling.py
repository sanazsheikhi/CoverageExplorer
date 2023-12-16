import numpy as np
from copy import copy, deepcopy
from scipy.spatial import ConvexHull



class Convex:

    def __init__(self):
        self.filtered_points = None
        self.bounding_max_y = None
        self.bounding_min_y = None
        self.bounding_max_x = None
        self.bounding_min_x = None
        self.bounding_rectangle = None


    def make_exact_convex_2(self, clustered_points, cluster_count):
        'Clusters the points and makes a convex hull out of each cluster'

        'Clustering Data Points'
        self.cluster_count = cluster_count
        self.clusters = clustered_points

        'Making convex and prepare for sampling'
        self.vols = []
        self.delns = []
        self.convexa = []
        self.dims = 2 #clustered_points[0].shape[1]
        self.hull_vertices = (np.array([], dtype=np.int64).
                              reshape(0, self.dims))
        self.cluster_hull_vertices = []
        self.cluster_hulls = []

        for cluster in self.clusters :
            cluster_points = cluster

            # Get the first two columns of the original array
            first_two_columns = cluster[:, :self.dims]
            # Reshape the 'first_two_columns' array to n*2 shape
            n_two_columns = first_two_columns.reshape(-1, self.dims)
            convex_hull = ConvexHull(n_two_columns, incremental=True)

            self.cluster_hulls.append(copy(convex_hull))
            self.convexa.append(convex_hull)
            hull = cluster_points[convex_hull.vertices, :self.dims]

            'To be used for point sampling'
            self.hull_vertices = np.vstack([self.hull_vertices, hull])
            self.cluster_hull_vertices.append(copy(hull))


        self.get_bounding_rectangle()
        self.current_convex = 0



    def update_convexHull(self, array_points):
        'Order of operations matter due to shared data structures'
        self.points = copy(np.append(self.points, array_points, axis=0))
        self.convexhull.add_points(array_points, restart=False)
        self.hull_vertices = copy(self.points[self.convexhull.vertices])
        self.get_bounding_rectangle()

    def get_bounds(self):
        return ([self.bounding_min_x, self.bounding_min_y],
                [self.bounding_max_x, self.bounding_max_y])


    def get_bounding_rectangle(self):
        'Outer approximation for the convexhull'
        hull_vertices = copy(self.hull_vertices)
        # Close the convex hull shape
        hull_vertices = np.vstack((hull_vertices, hull_vertices[0]))

        # Find the extreme points in each direction
        leftmost = np.argmin(hull_vertices[:, 0])
        rightmost = np.argmax(hull_vertices[:, 0])
        topmost = np.argmax(hull_vertices[:, 1])
        bottommost = np.argmin(hull_vertices[:, 1])

        # Get the extreme points
        leftmost_point = hull_vertices[leftmost]
        rightmost_point = hull_vertices[rightmost]
        topmost_point = hull_vertices[topmost]
        bottommost_point = hull_vertices[bottommost]

        # Calculate the bounding rectangle
        self.bounding_min_x = min_x = leftmost_point[0]
        self.bounding_max_x = max_x = rightmost_point[0]
        self.bounding_min_y = min_y = bottommost_point[1]
        self.bounding_max_y = max_y = topmost_point[1]

        # Construct the bounding rectangle
        self.bounding_rectangle = np.array([[min_x, min_y], [max_x, min_y],
                                            [max_x, max_y], [min_x, max_y],
                                            [min_x, min_y]])
        # print(f"get box bound  [{min_x},{min_y}], [{max_x},{max_y}]")


    def sample(self, num_samples=1):
        """
            Sampling points from the bounding rectangle but outside of the
            convex.The purpose is to sample points from vicinity of the
            training data
        """
        dim = 2
        sampled_points = np.random.uniform(low= [self.bounding_min_x,
                                                 self.bounding_min_y],
                                            high=[self.bounding_max_x,
                                                  self.bounding_max_y],
                                            size=(50*num_samples, dim))

        # Filter out points inside the convex hull
        filtered_points = []
        for point in sampled_points:
            if not self.point_inside_any_convex_hull(point,
                                                self.cluster_hull_vertices):
                filtered_points.append(point)
            if len(filtered_points) == num_samples:
                break
        self.filtered_points = np.array(filtered_points)
        # self.plot()
        return self.filtered_points


    def point_inside_any_convex_hull(self, point, convex_hulls):
        'Check if a point is inside any of the convex hulls'
        for hull in convex_hulls:
            if self.point_inside_polygon(point, hull):
                return True
        return False


    def point_inside_polygon(self, point, polygon):
        """Check if a point is inside a polygon
        via the ray-casting algorithm."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0, 0], polygon[0, 1]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n, 0], polygon[i % n, 1]
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = ((point[1] - p1y) *
                                       (p2x - p1x) / (p2y - p1y) + p1x)
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
