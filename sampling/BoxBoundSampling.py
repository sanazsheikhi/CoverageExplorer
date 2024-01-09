import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class BoxBoundSampling:

    min_x = 0.0
    max_x = 0.0
    min_y = 0.0
    max_y = 0.0
    upper_bound = None
    lower_bound = None
    kd_trees = None
    benchmark = ''
    min_distance = 10 # TBD based on case study
    max_distance = 100 # TBD based on case study


    def __init__(self, benchmark, trajectories):

        print(f"BoxBoundSampling")
        self.trajectories = self.convert_to_2D_trajectories(trajectories)
        self.calculate_bounding_box( self.trajectories)
        self.construct_kd_trees( self.trajectories)
        self.benchmark = benchmark


    def convert_to_2D_trajectories(self, trajectories_4D):
        trajectories_2D = []
        for traj in trajectories_4D:
            traj_2D = traj[:, :2]
            trajectories_2D.append(traj_2D) # array to a list
        return trajectories_2D


    def set_min_distance(self, min_dist):
        self.min_distance = min_dist


    def get_bounds(self):
        return self.lower_bound, self.upper_bound


    def calculate_bounding_box(self, trajectories):
        self.min_x = min(point[0] for traj in trajectories for point in traj)
        self.max_x = max(point[0] for traj in trajectories for point in traj)
        self.min_y = min(point[1] for traj in trajectories for point in traj)
        self.max_y = max(point[1] for traj in trajectories for point in traj)
        self.lower_bound = [self.min_x, self.min_y]
        self.upper_bound = [self.max_x, self.max_y]
        print(f"calculate_bounding_box: "
              f"low {self.lower_bound} high {self.upper_bound} ")


    def sample_point(self):
        ' sample_point_with_min_distance'
        while True:
            random_point = np.random.uniform(low=self.lower_bound,
                                             high=self.upper_bound, size=2)
            a = self.is_minimum_distance_satisfied(random_point, self.min_distance)
            # b = self.is_maximum_distance_satisfied(random_point, self.max_distance)
            if a:
                return random_point


    def construct_kd_trees(self, trajectories):
        self.kd_trees = []
        self.kd_trees_sizes = []
        for traj in trajectories:
            self.kd_trees.append(KDTree(traj))
            self.kd_trees_sizes.append(traj.shape[0])


    def is_maximum_distance_satisfied(self, point, max_distance_threshold):
        """
        This function finds the max distance of point from each kdtree,
        and if the point has max distance less than the threshold to at least
        one tree then it is accepted; meaning it is in proper distance to training data
        """
        for idx, kd_tree in enumerate(self.kd_trees):
            # Calculate distances to point P for all points in the KDTree
            distances, _ = kd_tree.query(point, k=self.kd_trees_sizes[idx])
            if max(distances) < max_distance_threshold:
                return  True
        return  False


    def is_minimum_distance_satisfied(self, point, min_distance):
        for kd_tree in self.kd_trees:
            _, idxs = kd_tree.query(point, k=1, distance_upper_bound=min_distance)
            closest_point_index = idxs
            if closest_point_index < kd_tree.data.shape[0]:
                closest_point = kd_tree.data[closest_point_index]
                distance = math.sqrt((point[0] - closest_point[0]) ** 2 + (point[1] - closest_point[1]) ** 2)
                if distance < min_distance:
                    return False
        return True


    def plot_trajectories_and_sampled_points(self, sampled_points=None):
        plt.figure(figsize=(8, 8))
        for traj in self.trajectories:
            x, y = zip(*traj)
            plt.plot(x, y, 'b-', alpha=0.7)

        # Plot bounding box
        plt.plot([self.min_x, self.max_x, self.max_x, self.min_x, self.min_x],
                 [self.min_y, self.min_y, self.max_y, self.max_y, self.min_y],
                 'r--', label='Bounding Box')

        # Plot sampled points (if provided)
        if sampled_points is not None:
            sampled_x, sampled_y = sampled_points[0], sampled_points[1]
            plt.scatter(sampled_x, sampled_y, color='g', marker='o', label='Sampled Points')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.title ='Trajectories and Sampled Points'
        plt.show()




