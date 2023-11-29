from scipy.optimize import linear_sum_assignment
import numpy as np

def calculate_euclidean_distance(pt1, pt2, scaling_factor=1):
    assert pt1.shape == pt2.shape, "points must be same dim."
    distance = np.sqrt(np.sum((pt1-pt2)**2))
    return distance * scaling_factor

def calculate_distance_matrix(arr1, arr2, scaling_factor=1):
    size_arr1 = arr1.shape[0]
    size_arr2 = arr2.shape[0]
    dist_matrix = np.zeros((size_arr1, size_arr2))
    for i in range(size_arr1):
        for j in range(size_arr2):
            dist_matrix[i, j] = calculate_euclidean_distance(arr1[i], arr2[j], scaling_factor)
    return dist_matrix 

def match_points(set1, set2):
    distance_matrix = calculate_distance_matrix(set1, set2)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    print(list(zip(row_ind, col_ind)))

    matched_pairs = [(set1[i], set2[j]) for i, j in zip(row_ind, col_ind) if distance_matrix[i, j]] 

    # Identify unmatched points in set1
    unmatched_set1_indices = set(range(len(set1))) - set(row_ind)
    unmatched_points_set1 = [set1[i] for i in unmatched_set1_indices]

    # Identify unmatched points in set2
    unmatched_set2_indices = set(range(len(set2))) - set(col_ind)
    unmatched_points_set2 = [set2[i] for i in unmatched_set2_indices]

    return matched_pairs, unmatched_points_set1, unmatched_points_set2


if __name__ == '__main__':
    predictions = np.array([
        [104, 101], 
        [91, 89],
        [0, 0]
    ])
    gt = np.array([
        [100, 100], 
        [90, 90]
    ])

    matched_pairs, unmatched_predictions, unmatched_gt = match_points(predictions, gt)
    print('matched', matched_pairs)
    print('unmatched preds (false positives)', unmatched_predictions)
    print('unmatched gt (false negatives)', unmatched_gt)

    predictions = np.array([
        [104, 101], 
        [91, 89],
    ])
    gt = np.array([
        [100, 100], 
        [90, 90],
        [0, 0]
    ])

    matched_pairs, unmatched_predictions, unmatched_gt = match_points(predictions, gt)
    print('matched', matched_pairs)
    print('unmatched preds (false positives)', unmatched_predictions)
    print('unmatched gt (false negatives)', unmatched_gt)
