# WFG_RS

A Rust [ndarray](https://crates.io/crates/ndarray)-based implementation of the While-Bradstreet-Barone algorithm to compute the hypervolume indicator for multi-objective optimization.

_We consider the minimization of each objectives._

## Main function

The function `wfg` takes
- `front`: A 2D array of floats (num_points, num_objectives).
- `ref_point`: A 1D array of floats corresponding to a reference point. This point can be computed via the `reference_point` function.
- `lexsorted`: A boolean flag. If `true` then we assume the `front` is lexigographically sorted. If `false` it will lex-sort `front`.
- `extract_front`: A boolean flag. If `true` extract the actual Pareto set from the inputs. If `false`, assumes the input is a Pareto set.

## Example

```rust
use ndarray::array;
use wfg_rs::*;

// 4 points and 3 objectives
let points = array![
    [1.0, 4.0, 4.0],
    [2.0, 2.0, 3.0],
    [3.0, 1.5, 2.0],
    [4.0, 1.0, 1.0],
];
let ref_point = reference_point(points.view());
assert_eq!(ref_point, array![4.0, 4.0, 4.0]);

// point, reference, assume lexicographically sorted, extract Pareto set from inputs
let hv = wfg::<f64>(points.view(), ref_point.view(), true, false);
assert_eq!(hv, 7.0);
```

## Bibliography

The implementation of the WFG algorithm is adapted from [A Fast Way of Calculating Exact Hypervolumes](https://ieeexplore.ieee.org/document/5766730).

## Dependencies
- [ndarray](https://crates.io/crates/ndarray): `version = "0.17.2", default-features = false`
- [num](https://docs.rs/num/latest/num/): `version = "0.4.3", default-features = false, features = ["libm"] `

## License

**MIT**