use ndarray::{ArrayView2, s};
use num::Float;

/// Returns the indices of the non-dominated points in the lexicographically sorted input array.
///
/// # Arguments
/// * `values` - A 2D [`Float`] array of shape (n_points, n_objectives) assuming lexicographically sorted.
///
/// # Returns
/// A vector of indices of the non-dominated points in the input array.
///
/// # Notes
///
/// See [`lexsort`](crate::sort::lexsort) for lexicographic sorting of the input array.
pub fn argfront_lexsorted<F: Float>(values: ArrayView2<F>) -> Vec<usize> {
    let values = values.slice(s![.., 1..]);
    let mut on_front = Vec::new();

    let mut remaining_idx = (0..values.nrows()).collect::<Vec<usize>>();
    let mut new_len;

    while !remaining_idx.is_empty() {
        let non_dominated_idx = remaining_idx[0];
        on_front.push(non_dominated_idx);

        let top_row = values.row(non_dominated_idx);
        new_len = 0;

        for i in 0..remaining_idx.len() {
            let idx = remaining_idx[i];
            let dominates = values
                .row(idx)
                .iter()
                .zip(top_row.iter())
                .any(|(v1, v2)| v1 < v2);
            if dominates {
                remaining_idx[new_len] = idx;
                new_len += 1;
            }
        }
        remaining_idx.truncate(new_len);
    }
    on_front
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    #[test]
    fn test_lexsort() {
        let points = array![
            [1.0, 4.0, 4.0],
            [2.0, 1.0, 3.0],
            [2.0, 2.0, 3.0],
            [3.0, 1.5, 2.0],
            [3.0, 3.0, 2.0],
            [4.0, 1.0, 1.0],
        ];
        let true_ground = array![
            [1.0, 4.0, 4.0],
            [2.0, 1.0, 3.0],
            [3.0, 1.5, 2.0],
            [4.0, 1.0, 1.0],
        ];
        let mut on_front = super::argfront_lexsorted(points.view());
        on_front.sort();
        let true_idx = vec![0, 1, 3, 5];
        assert_eq!(on_front, true_idx, "Front indices extraction failed");
        let front = points.select(ndarray::Axis(0), &on_front);
        assert_eq!(front, true_ground, "Front extraction failed");
    }
}
