#[no_mangle]
pub unsafe extern "C" fn update_h_interior(
    hx: *mut f64,
    ey: *const f64,
    mhx: *const f64,
    nz: usize,
    dz: f64,
) {
    let hx = std::slice::from_raw_parts_mut(hx, nz);
    let ey = std::slice::from_raw_parts(ey, nz);
    let mhx = std::slice::from_raw_parts(mhx, nz);

    for i in 0..(nz - 1) {
        hx[i] += mhx[i] * (ey[i + 1] - ey[i]) / dz;
    }
}

#[no_mangle]
pub unsafe extern "C" fn update_e_interior(
    ey: *mut f64,
    hx: *const f64,
    mey: *const f64,
    nz: usize,
    dz: f64,
) {
    let ey = std::slice::from_raw_parts_mut(ey, nz);
    let hx = std::slice::from_raw_parts(hx, nz);
    let mey = std::slice::from_raw_parts(mey, nz);

    for i in 1..nz {
        ey[i] += mey[i] * (hx[i] - hx[i - 1]) / dz;
    }
}
