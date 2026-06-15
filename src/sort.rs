use ndarray::{Array2, ArrayView2, Axis};
use num::Float;
use core::cmp::Ordering;

/// Lexicographically sorts the input objects in ascending order of their objectives.
///
/// # Arguments
/// * `input` - A 2D [`Float`] array of shape (n_points, n_objectives).
///
/// # Returns
/// A lexicographically sorted 2D [`Float`] array of shape (n_points, n_objectives)
///
/// # Notes
///
/// It uses `partial_cmp` to compare the objective values, which means that it will panic if any of the values is NaN.
pub fn lexsort<F: Float>(input: ArrayView2<F>) -> Array2<F> {
    let nrows = input.nrows();
    let ncols = input.ncols();

    let mut indices: Vec<usize> = (0..nrows).collect();
    indices.sort_by(|&a, &b| {
        for i in 0..ncols {
            let x = input[[a, i]];
            let y = input[[b, i]];
            if x < y {
                return Ordering::Less;
            }
            if x > y {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    });
    input.select(Axis(0), &indices)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    #[test]
    fn test_lexsort() {
        let points = array![
            [3.0, 3.0, 2.0],
            [1.0, 4.0, 4.0],
            [2.0, 1.0, 3.0],
            [4.0, 1.0, 1.0],
            [3.0, 1.5, 2.0],
            [2.0, 2.0, 3.0],
        ];
        let sorted = super::lexsort(points.view());
        let true_ground = array![
            [1.0, 4.0, 4.0],
            [2.0, 1.0, 3.0],
            [2.0, 2.0, 3.0],
            [3.0, 1.5, 2.0],
            [3.0, 3.0, 2.0],
            [4.0, 1.0, 1.0],
        ];
        assert_eq!(sorted, true_ground, "Lexicographic sorting failed");
    }
}
