use crate::{pareto::argfront_lexsorted, sort::lexsort};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use num::Float;
use std::{fmt::Debug, ops::Sub};

/// Computes the inclusive hypervolumes of a set of points in the objective space with respect to a reference point.
///
/// See eq. 2 from [WFG](https://ieeexplore.ieee.org/document/5766730), i.e. the hyper-rectangle bounded by a point from the input set and the reference point.
///
/// # Arguments
/// * `set` - A 2D [`Float`] array of shape (n_points, n_objectives).
/// * `ref_point` - A 1D [`Float`] array of shape (n_objectives); i.e. the reference point.
///
/// # Returns
/// A 1D [`Float`] array of shape (n_points) containing the inclusive hypervolumes of each point in the input set with respect to the reference point.
pub fn inclusive_wfg<F: Float + 'static>(
    set: ArrayView2<F>,
    ref_point: ArrayView1<F>,
) -> Array1<F> {
    ref_point.sub(&set).product_axis(Axis(1))
}

/// Computes the exclusive hypervolume of a point p of a given inclusive hypervolume with respect to a Pareto set and reference point.
///
/// See eq. 1 from [WFG](https://ieeexplore.ieee.org/document/5766730), i.e. the area E from Fig.1.
///
/// # Arguments
/// * `front` - A 2D [`Float`] array of shape (n_points, n_objectives) representing the Pareto set lexicographically sorted.
/// * `inclusive` - A [`Float`] value representing the inclusive hypervolume.
/// * `ref_point` - A 1D [`Float`] array of shape (n_objectives); i.e. the reference point.
///
/// # Returns
/// A [`Float`] value representing the exclusive hypervolume.
pub fn exclusive_wfg<F: Float + 'static>(
    front: ArrayView2<F>,
    inclusive: F,
    ref_point: ArrayView1<F>,
) -> F {
    let on_front = argfront_lexsorted(front);
    inclusive - _wfg(front.select(Axis(0), &on_front).view(), ref_point)
}

/// Limits the points in the input set with respect to a pivot point.
///
/// See eq. 6 and 7 from [WFG](https://ieeexplore.ieee.org/document/5766730).
///
/// # Arguments
/// * `inputs` - A 2D [`Float`] array of shape (n_points, n_objectives).
/// * `pivot` - A 1D [`Float`] array of shape (n_objectives).
///
/// # Returns
/// A 2D [`Float`] array of shape (n_points, n_objectives) containing the points from inputs limited by the pivot.
pub fn limits_wfg<F: Float + 'static>(inputs: ArrayView2<F>, pivot: ArrayView1<F>) -> Array2<F> {
    let n_rows = inputs.nrows();
    let n_cols = inputs.ncols();
    let mut limited = Array2::<F>::uninit((n_rows, n_cols));

    Zip::from(limited.rows_mut())
        .and(inputs.rows())
        .for_each(|mut out_row, in_row| {
            Zip::from(&mut out_row)
                .and(&in_row)
                .and(&pivot)
                .for_each(|o, &a, &b| {
                    o.write(a.max(b));
                });
        });
    unsafe { limited.assume_init() }
}

/// Computes the hypervolume of a Pareto set with respect to a reference point using the WFG algorithm.
///
/// See [WFG](https://ieeexplore.ieee.org/document/5766730) for details on the algorithm.
///
/// # Arguments
/// * `front` - A 2D [`Float`] array of shape (n_points, n_objectives) representing the lexicographically sorted Pareto set.
/// * `ref_point` - A 1D [`Float`] array of shape (n_objectives) representing the reference point.
///
/// # Returns
/// A [`Float`] value representing the hypervolume indicator of the input Pareto set.
pub fn _wfg<F: Float + 'static>(front: ArrayView2<F>, ref_point: ArrayView1<F>) -> F {
    if front.nrows() == 0 {
        return F::zero();
    }
    if !front.is_all_infinite() {
        return F::infinity();
    }

    let inclusive = inclusive_wfg(front, ref_point);
    let mut sum = F::zero();
    for i in 0..front.nrows() - 1 {
        let pivot = front.slice(s![i, ..]);
        let others = front.slice(s![i + 1.., ..]);
        let limited = limits_wfg(others, pivot);
        let excl = exclusive_wfg(limited.view(), inclusive[i], ref_point);
        sum = sum.add(excl);
    }
    inclusive[front.nrows() - 1] + sum
}

/// Computes the hypervolume indicator of a Pareto set with respect to a reference point using the WFG algorithm.
///
/// This function is a wrapper around the internal [`_wfg`] function that ensures the input Pareto set is lexicographically sorted before calling the internal function.
///
/// # Arguments
/// * `inputs` - A 2D [`Float`] array of shape (n_points, n_objectives) representing the input set.
/// * `ref_point` - A 1D [`Float`] array of shape (n_objectives) representing the reference point.
/// * `lexsorted` - A boolean flag indicating whether the input Pareto set is already lexicographically sorted.
/// * `extract_front` - A boolean flag indicating whether to extract the Pareto front before computing the hypervolume.
///
/// # Example
/// 
/// ```
/// // 4 points and 3 objectives
/// let points = array![
///     [1.0, 4.0, 4.0],
///     [2.0, 2.0, 3.0],
///     [3.0, 1.5, 2.0],
///     [4.0, 1.0, 1.0],
/// ];
/// let ref_point = reference_point(points.view());
/// assert_eq!(ref_point, array![4.0, 4.0, 4.0]);
/// 
/// // point, reference, assume lexicographically sorted, extract Pareto set from inputs
/// let hv = wfg::<f64>(points.view(), ref_point.view(), true, false);
/// assert_eq!(hv, 7.0);
/// ```
pub fn wfg<F: Float + Debug + 'static>(
    inputs: ArrayView2<F>,
    ref_point: ArrayView1<F>,
    lexsorted: bool,
    extract_front: bool,
) -> F {
    if extract_front {
        if !lexsorted {
            let sorted = lexsort(inputs);
            let on_front = argfront_lexsorted(sorted.view());
            _wfg(sorted.select(Axis(0), &on_front).view(), ref_point.view())
        } else {
            let on_front = argfront_lexsorted(inputs.view());
            _wfg(inputs.select(Axis(0), &on_front).view(), ref_point.view())
        }
    } else if !lexsorted {
        let sorted = lexsort(inputs);
        _wfg(sorted.view(), ref_point.view())
    } else {
        _wfg(inputs.view(), ref_point.view())
    }
}

/// Computes the reference point of a set for the hypervolume indicator.
/// Here, it is the point with the maximum value in each objective dimension.
pub fn reference_point<F: Float + 'static>(set: ArrayView2<F>) -> Array1<F> {
    let mut ref_point = Array1::<F>::zeros(set.ncols());
    Zip::from(ref_point.view_mut())
        .and(set.columns())
        .for_each(|ref_val, col| {
            *ref_val = col.fold(F::neg_infinity(), |a, &b| a.max(b));
        });
    ref_point
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::wfg::{inclusive_wfg, reference_point, wfg};

    #[test]
    fn test_ref_point() {
        let points = array![
            [1.0, 4.0, 4.0],
            [2.0, 2.0, 3.0],
            [3.0, 1.5, 2.0],
            [4.0, 1.0, 1.0],
        ];
        let ref_point = reference_point(points.view());
        assert_eq!(
            ref_point,
            array![4.0, 4.0, 4.0],
            "Reference point is wrong."
        );
    }

    #[test]
    fn test_wfg_incl_hypervolume() {
        let ref_point = array![4.0, 4.0, 4.0];
        let points = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0],];
        let incl_hv = inclusive_wfg(points.view(), ref_point.view());
        let true_ground = array![27.0, 8.0, 1.0];
        assert_eq!(incl_hv, true_ground, "Inclusive hypervolume is wrong.");
    }

    #[test]
    fn test_wfg_hypervolume_lexsorted() {
        let points = array![
            [1.0, 4.0, 4.0],
            [2.0, 2.0, 3.0],
            [3.0, 1.5, 2.0],
            [4.0, 1.0, 1.0],
        ];
        let ref_point = reference_point(points.view());
        assert_eq!(
            ref_point,
            array![4.0, 4.0, 4.0],
            "Reference point is wrong."
        );

        let hv = wfg(points.view(), ref_point.view(), true, false);
        assert_eq!(hv, 7.0, "WFG hypervolume is wrong.");
    }

    #[test]
    fn test_wfg_hypervolume_not_lexsorted() {
        let points = array![
            [2.0, 2.0, 3.0],
            [4.0, 1.0, 1.0],
            [1.0, 4.0, 4.0],
            [3.0, 1.5, 2.0],
        ];
        let ref_point = reference_point(points.view());
        assert_eq!(
            ref_point,
            array![4.0, 4.0, 4.0],
            "Reference point is wrong."
        );

        let hv = wfg(points.view(), ref_point.view(), false, false);
        assert_eq!(hv, 7.0, "WFG hypervolume is wrong.");
    }

    #[test]
    fn test_wfg_hypervolume_unordered_many() {
        let points = array![
            [
                4.26225466e-01,
                6.50878057e-01,
                6.32992389e-01,
                1.17652532e-01
            ],
            [
                6.49630742e-01,
                1.47810456e-01,
                6.18220458e-01,
                5.09327642e-01
            ],
            [
                7.88959899e-01,
                4.26108249e-01,
                5.28124747e-01,
                4.54778507e-01
            ],
            [
                9.25598652e-01,
                1.56702835e-01,
                4.22865400e-02,
                3.03672560e-01
            ],
            [
                2.77748898e-01,
                7.01537616e-01,
                9.12713531e-02,
                7.61665414e-01
            ],
            [
                4.23176112e-01,
                5.60970719e-01,
                4.64610056e-02,
                7.56151094e-01
            ],
            [
                1.06028446e-01,
                9.49482845e-01,
                6.78652553e-01,
                8.38704804e-01
            ],
            [
                7.90543159e-01,
                8.46030022e-02,
                8.15726396e-01,
                6.06057103e-01
            ],
            [
                5.99999858e-01,
                8.22654582e-01,
                7.28591904e-01,
                8.48238078e-01
            ],
            [
                6.08915139e-01,
                8.95856726e-01,
                4.86564346e-01,
                8.14470287e-01
            ],
            [
                5.74027879e-01,
                2.87900617e-01,
                4.83968955e-01,
                6.00291561e-01
            ],
            [
                1.77774375e-01,
                5.97149708e-02,
                5.31856169e-01,
                2.03102185e-01
            ],
            [
                4.13059605e-01,
                2.27272245e-01,
                2.12780575e-01,
                8.67907015e-01
            ],
            [
                8.41018401e-01,
                1.30208479e-01,
                4.34692570e-01,
                4.73857890e-02
            ],
            [
                2.45305629e-01,
                7.18724867e-01,
                8.54038335e-01,
                5.63435511e-01
            ],
            [
                2.66656803e-01,
                6.85716182e-01,
                5.39709855e-01,
                5.73027093e-01
            ],
            [
                7.03487277e-01,
                3.64993361e-01,
                2.08325979e-01,
                1.34185331e-02
            ],
            [
                3.49126555e-01,
                7.37295214e-01,
                4.69980837e-01,
                9.11173490e-01
            ],
            [
                5.48343169e-01,
                7.39964937e-01,
                6.33717446e-01,
                6.10842593e-01
            ],
            [
                9.99390947e-01,
                6.97960923e-01,
                6.08997417e-01,
                1.94035033e-02
            ],
            [
                7.76903843e-01,
                3.61159448e-01,
                6.07154265e-01,
                7.09692015e-01
            ],
            [
                4.11218338e-01,
                8.18167392e-01,
                4.54270358e-01,
                7.73338314e-01
            ],
            [
                5.95474451e-01,
                1.39633422e-01,
                2.62785582e-01,
                4.80530080e-01
            ],
            [
                3.86734534e-01,
                9.64929753e-01,
                8.30395924e-01,
                2.71793822e-01
            ],
            [
                4.55521736e-01,
                6.04561383e-01,
                7.41375993e-01,
                5.95949704e-01
            ],
            [
                1.67068998e-01,
                2.06250060e-01,
                9.08683591e-01,
                4.39607470e-01
            ],
            [
                4.27617850e-01,
                2.82429740e-01,
                9.22289357e-01,
                5.56905270e-01
            ],
            [
                4.32667445e-01,
                2.34081614e-01,
                4.26821297e-01,
                7.95216595e-01
            ],
            [
                4.00757312e-01,
                4.00768874e-01,
                7.71266996e-01,
                6.69704347e-01
            ],
            [
                3.68918136e-01,
                7.34327084e-01,
                1.04371199e-02,
                6.82828636e-01
            ],
            [
                6.01836935e-01,
                5.40944650e-01,
                7.02757899e-01,
                9.33764106e-01
            ],
            [
                6.55833719e-01,
                7.35293416e-01,
                8.60517360e-01,
                3.52141253e-01
            ],
            [
                1.55426661e-02,
                2.18394404e-01,
                2.63901888e-01,
                4.89723089e-01
            ],
            [
                4.00492412e-01,
                2.58923829e-01,
                6.47011284e-01,
                3.23584695e-01
            ],
            [
                7.68327231e-02,
                4.29068496e-01,
                7.00251566e-01,
                5.86087810e-01
            ],
            [
                9.69047886e-01,
                4.83868255e-01,
                5.88312849e-01,
                8.63668332e-01
            ],
            [
                9.28182825e-01,
                9.21894613e-01,
                7.26190451e-01,
                5.84298995e-01
            ],
            [
                7.06509469e-03,
                7.32444356e-03,
                3.77074056e-01,
                9.84368617e-01
            ],
            [
                2.90968625e-01,
                8.46023586e-01,
                1.30291515e-01,
                6.68017030e-01
            ],
            [
                5.75454039e-01,
                3.04118877e-01,
                7.75004849e-01,
                8.83592970e-01
            ],
            [
                7.63077260e-01,
                6.58443585e-01,
                2.64742129e-01,
                3.37331301e-01
            ],
            [
                1.09328993e-01,
                5.37218410e-01,
                4.56396621e-01,
                4.94071330e-01
            ],
            [
                4.29631973e-01,
                5.26956573e-01,
                6.48917222e-01,
                5.27554333e-01
            ],
            [
                4.17237604e-01,
                3.68521600e-01,
                3.57279078e-02,
                4.21277361e-01
            ],
            [
                1.17357125e-01,
                1.13931766e-01,
                8.40028081e-01,
                6.37487670e-01
            ],
            [
                8.58770603e-01,
                4.16031879e-02,
                3.88636696e-02,
                4.34004578e-01
            ],
            [
                4.72178133e-01,
                6.97718283e-01,
                1.91356835e-01,
                5.76211945e-01
            ],
            [
                7.50525511e-01,
                8.91081445e-01,
                1.00158955e-01,
                3.38361872e-01
            ],
            [
                8.26161474e-01,
                8.12377227e-01,
                9.84261569e-01,
                7.01862925e-01
            ],
            [
                9.30439473e-01,
                5.07950263e-01,
                1.00159952e-01,
                1.72027528e-01
            ],
            [
                7.64722163e-01,
                3.66456642e-01,
                9.29885002e-01,
                6.84854798e-01
            ],
            [
                4.54331143e-01,
                5.93950877e-01,
                4.21324256e-02,
                5.12072233e-01
            ],
            [
                5.92436224e-01,
                7.98899151e-01,
                1.64151436e-01,
                2.49418020e-01
            ],
            [
                5.18041399e-01,
                6.89208407e-01,
                2.95905667e-01,
                3.85363860e-01
            ],
            [
                5.14995232e-01,
                1.38425028e-01,
                5.86283036e-01,
                8.42450258e-01
            ],
            [
                9.06766397e-01,
                1.98638976e-01,
                4.32770063e-01,
                4.81815988e-01
            ],
            [
                7.30922844e-01,
                1.24596583e-01,
                3.38937187e-02,
                2.83297663e-01
            ],
            [
                6.46205647e-01,
                4.06338079e-01,
                9.00989556e-01,
                8.79860487e-01
            ],
            [
                2.80119121e-01,
                9.57577243e-01,
                8.55759738e-01,
                1.64238546e-01
            ],
            [
                7.08519586e-01,
                4.62588644e-01,
                2.04728665e-01,
                1.04498871e-01
            ],
            [
                1.95770372e-01,
                3.49358038e-01,
                8.13439631e-02,
                8.50365226e-01
            ],
            [
                6.95859049e-01,
                5.23176544e-01,
                5.36460591e-01,
                1.47418490e-01
            ],
            [
                6.62104160e-02,
                6.20962325e-01,
                9.99023804e-01,
                3.51763862e-01
            ],
            [
                3.45824949e-01,
                3.01945001e-01,
                8.40882493e-01,
                8.14712444e-01
            ],
            [
                1.52462778e-01,
                6.00029331e-01,
                3.02635745e-01,
                4.75153674e-01
            ],
            [
                1.43068419e-01,
                2.77679451e-01,
                7.12388251e-01,
                6.62236348e-01
            ],
            [
                9.58567318e-01,
                6.41506562e-01,
                2.24702525e-02,
                7.53093126e-02
            ],
            [
                5.74811563e-01,
                2.27407231e-01,
                9.79436294e-01,
                3.77878341e-02
            ],
            [
                1.52896726e-01,
                9.29681773e-01,
                2.94481280e-01,
                7.10388298e-01
            ],
            [
                4.97359810e-01,
                9.07787077e-01,
                5.06168716e-01,
                9.46407419e-02
            ],
            [
                1.15280516e-01,
                5.20124508e-01,
                5.32895977e-01,
                2.78259528e-01
            ],
            [
                4.15800463e-01,
                1.87639237e-01,
                8.06856278e-01,
                9.46361220e-01
            ],
            [
                2.25082557e-01,
                7.77909204e-01,
                2.51836539e-01,
                9.44536170e-01
            ],
            [
                3.72795681e-01,
                8.34331201e-01,
                5.00620958e-01,
                5.69343199e-01
            ],
            [
                6.16885197e-04,
                2.80816785e-01,
                3.35837384e-01,
                8.30767056e-01
            ],
            [
                7.78527540e-01,
                9.78795896e-01,
                1.38302134e-01,
                7.12601187e-01
            ],
            [
                5.10080961e-01,
                2.45655969e-01,
                2.76742907e-01,
                6.16217904e-01
            ],
            [
                7.46902586e-01,
                4.63766746e-01,
                1.00531995e-01,
                9.14171028e-01
            ],
            [
                7.07830421e-01,
                8.14744240e-01,
                1.61980851e-01,
                6.21353173e-01
            ],
            [
                4.54846974e-01,
                6.27966961e-02,
                7.59110427e-01,
                5.42247240e-01
            ],
            [
                2.73414733e-01,
                7.29813402e-01,
                1.33060427e-01,
                5.71265539e-01
            ],
            [
                7.84700680e-01,
                2.42101430e-01,
                2.71356058e-02,
                9.11978675e-01
            ],
            [
                3.48925450e-01,
                1.99169401e-01,
                8.41328635e-01,
                2.47912101e-01
            ],
            [
                9.72988322e-01,
                5.47281774e-01,
                4.66462795e-01,
                2.36487172e-01
            ],
            [
                6.25684896e-01,
                2.59548538e-01,
                2.63911917e-01,
                7.42596595e-01
            ],
            [
                4.74486785e-01,
                5.33621370e-01,
                9.59243701e-01,
                1.28862936e-01
            ],
            [
                5.77618945e-01,
                4.09794987e-01,
                5.08825162e-01,
                8.98492734e-01
            ],
            [
                7.95365281e-01,
                8.18658470e-01,
                3.65239062e-01,
                8.17209781e-01
            ],
            [
                7.69340362e-01,
                3.12892880e-01,
                6.09930873e-01,
                5.45668311e-01
            ],
            [
                9.15638063e-02,
                7.15302882e-01,
                5.91631415e-01,
                5.91536251e-01
            ],
            [
                3.30309224e-01,
                7.16547088e-01,
                4.87074758e-01,
                3.58631596e-01
            ],
            [
                9.40165870e-01,
                1.75503467e-01,
                5.08064035e-01,
                1.73382286e-01
            ],
            [
                8.36682355e-01,
                2.50414300e-01,
                6.36003873e-01,
                5.83859458e-01
            ],
            [
                7.35969467e-01,
                6.99717980e-01,
                6.30675898e-01,
                9.57175138e-01
            ],
            [
                6.93633813e-01,
                2.72450172e-01,
                5.01758661e-01,
                6.07905820e-01
            ],
            [
                8.80510136e-01,
                1.12584239e-01,
                8.07924017e-01,
                6.17075253e-01
            ],
            [
                7.58503903e-01,
                6.31839567e-01,
                1.97365495e-02,
                4.90389119e-02
            ],
            [
                4.24668899e-01,
                2.54556561e-01,
                7.22406708e-01,
                8.26293603e-01
            ],
            [
                5.01932715e-02,
                7.45281395e-02,
                1.83626891e-01,
                4.32207594e-01
            ],
            [
                9.75453990e-01,
                9.53995918e-01,
                9.50256694e-01,
                4.94360948e-01
            ]
        ];
        let ref_point = reference_point(points.view());
        let ref_true_ground = array![0.999390947, 0.978795896, 0.999023804, 0.984368617];
        assert_eq!(
            ref_point, ref_true_ground,
            "Reference point is wrong for many many points."
        );

        let hv = wfg(points.view(), ref_point.view(), false, true);
        assert!((hv - 0.5883493015142789_f64).abs() < f64::EPSILON, "WFG hypervolume is wrong.");
    }
}
