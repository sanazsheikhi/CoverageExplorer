import numpy as np
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import directed_hausdorff


class Clustering:

    def __init__(self):
        pass

    def calculate_trajectory_similarity(self, trajectory1, trajectory2):
        # Calculate similarity between two trajectories
        similarity = np.mean(np.linalg.norm(trajectory1 - trajectory2, axis=1))
        return similarity


    def kmeans_clustering(self, trajectories, k):
        num_trajectories = len(trajectories)
        trajectories = np.array(trajectories)

        similarity_matrix = np.zeros((num_trajectories, num_trajectories))
        for i in range(num_trajectories):
            for j in range(i, num_trajectories):
                similarity = self.calculate_trajectory_similarity(trajectories[i],
                                                                  trajectories[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # Perform K-means clustering on similarity matrix
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(similarity_matrix)
        labels = kmeans.labels_

        return labels


    def calculate_distance(self, traj1, traj2):
        return directed_hausdorff(traj1[:, :2], traj2[:, :2])[0]


    def find_most_different_trajectories(self, trajectories, percent):
        if len(trajectories) == 1:
            return trajectories, [0]

        distances = []
        M = len(trajectories)
        selected_index = []
        visited = [0] * M
        # Calculate pairwise distances between trajectories
        for i in range(M):
            for j in range(i + 1, M):
                distance = self.calculate_distance(trajectories[i],
                                                   trajectories[j])
                distances.append((i, j, distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[2])

        """ 
            Select the N most different trajectories as the items 
            being added to the set selected_trajectories are list and 
            immutable/have different hash, the set does not treat them as 
            same objects and adds duplicate items. We handle uniqness 
            of items by visited flags.
        """

        selected_trajectories = set()
        for i, j, distance in distances:
            if visited[i] == 0:
                selected_trajectories.add(i)
                selected_index.append(i)
                visited[i] = 1
            if visited[j] == 0:
                selected_trajectories.add(j)
                selected_index.append(j)
                visited[j] = 1
            if len(selected_trajectories) >= round(percent * len(trajectories)) + 1:
                break

        # Convert the selected trajectory indices to actual trajectories
        selected_trajectories = [trajectories[i] for i in selected_trajectories]

        return selected_trajectories, selected_index


    def select_trajectories(self, trajectories, percent):
        selected_trajectories = []
        if len(trajectories) == 1:
            return trajectories, [0]

        count = round(percent * len(trajectories))
        if count < len(trajectories)-1:
            count += 1

        selected_indice = random.sample(range(0, len(trajectories)), count)

        for i in selected_indice:
            selected_trajectories.append(trajectories[i])

        return  selected_trajectories, selected_indice


    def select_trajectories_Kmeans(self, trajectories, k, percent):

        'Clustering'
        labels = self.kmeans_clustering(trajectories, k)
        cluster_points = [] # points in clusters for convex and sampling
        clusters = max(labels) + 1

        'Choosing trajectories from clusters'
        selected_trajectories = []
        ls_st = []  # list of indices of selected trajectories

        for c in range(clusters):
            current_cluster_traj = []

            'list of indice of trajectories from main list that are in c'
            current_ls_st = []

            'getting trajectories that belong to each cluster'
            for i, value in enumerate(labels):
                if value == c:
                    current_cluster_traj.append(trajectories[i])
                    current_ls_st.append(i)

            mdt, selected_index = (self.find_most_different_trajectories
                                   (current_cluster_traj, percent=percent))

            #'randomly choose trajectories for each cluster'
            # mdt, selected_index = self.select_trajectories
            # (current_cluster_traj, percent=percent)

            selected_trajectories = selected_trajectories + mdt

            for i in selected_index:
                'indice of trajectories that got selected'
                ls_st.append(current_ls_st[i])

            cp = np.vstack(mdt)
            cluster_points.append(cp)

        return cluster_points, ls_st
