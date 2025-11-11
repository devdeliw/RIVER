/// Stores spacings between adjacent x_eval points
pub(crate) fn spacings(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut h = Vec::with_capacity(n - 1);

    for i in 0..n - 1 {
        h.push(x[i+1] - x[i]);
    }

    h
}


/// Prepares L/U matrices in col-major for CORAL TRSV
pub(crate) fn lu_to_dense_for_trsv(
    l_sub : &[f64],
    u_diag: &[f64], 
    c_sup : &[f64]
) -> (Vec<f64>, Vec<f64>, usize) {
    let n = u_diag.len();
    let lda = n;

    let mut l = vec![0.0; n * n];
    let mut u = vec![0.0; n * n];

    for j in 0..n {
        l[j * lda + j] = 1.0;
        if j + 1 < n {
            l[j * lda + (j + 1)] = l_sub[j+1];
        }
        u[j * lda + j] = u_diag[j];
        if j > 0 {
            u[j * lda + (j - 1)] = c_sup[j-1];
        }
    }

    (l, u, lda)
}


pub(crate) fn coeffs(
    n: usize,
    h: &[f64],
    y: &[f64],
    c_full: &[f64]
) -> (Vec<f64>, Vec<f64>) {
    // per-interval 
    // b_i, d_i (a_i is y[i], c_i = c_full[i])
    let mut bcoef = vec![0.0; n-1];
    let mut dcoef = vec![0.0; n-1];

    for i in 0..n - 1 {
        bcoef[i] = (y[i+1] - y[i]) / h[i] - (h[i] * (2.0 * c_full[i] + c_full[i+1])) / 3.0;
        dcoef[i] = (c_full[i+1] - c_full[i]) / (3.0 * h[i]);
    }

    (bcoef, dcoef)
}


pub(crate) fn find_interval(x: &[f64], xq: f64) -> usize {
    let n = x.len();
    let mut lo = 0;
    let mut hi = n - 1;

    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if x[mid] <= xq {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    lo
}


pub(crate) fn deltas(y: &[f64], h: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut d = Vec::with_capacity(n - 1);
    for i in 0..n - 1 { d.push((y[i+1] - y[i]) / h[i]); }
    d
}

