#[inline]
fn idx(i: usize, j: usize, ny: usize) -> usize {
    i * ny + j
}

#[no_mangle]
pub unsafe extern "C" fn calculate_curl_e_hz(
    ex: *const f64,
    ey: *const f64,
    d_ex_y: *mut f64,
    d_ey_x: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    per_x: i32,
    per_y: i32,
) {
    let ex = std::slice::from_raw_parts(ex, nx * ny);
    let ey = std::slice::from_raw_parts(ey, nx * ny);
    let d_ex_y = std::slice::from_raw_parts_mut(d_ex_y, nx * ny);
    let d_ey_x = std::slice::from_raw_parts_mut(d_ey_x, nx * ny);

    for i in 0..nx {
        for j in 0..(ny - 1) {
            d_ex_y[idx(i, j, ny)] = (ex[idx(i, j + 1, ny)] - ex[idx(i, j, ny)]) / dy;
        }
        d_ex_y[idx(i, ny - 1, ny)] = if per_y != 0 {
            (ex[idx(i, 0, ny)] - ex[idx(i, ny - 1, ny)]) / dy
        } else {
            -ex[idx(i, ny - 1, ny)] / dy
        };
    }

    for j in 0..ny {
        for i in 0..(nx - 1) {
            d_ey_x[idx(i, j, ny)] = (ey[idx(i + 1, j, ny)] - ey[idx(i, j, ny)]) / dx;
        }
        d_ey_x[idx(nx - 1, j, ny)] = if per_x != 0 {
            (ey[idx(0, j, ny)] - ey[idx(nx - 1, j, ny)]) / dx
        } else {
            -ey[idx(nx - 1, j, ny)] / dx
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn calculate_curl_h_hz(
    hz: *const f64,
    d_hz_y: *mut f64,
    d_hz_x: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    per_x: i32,
    per_y: i32,
) {
    let hz = std::slice::from_raw_parts(hz, nx * ny);
    let d_hz_y = std::slice::from_raw_parts_mut(d_hz_y, nx * ny);
    let d_hz_x = std::slice::from_raw_parts_mut(d_hz_x, nx * ny);

    for i in 0..nx {
        for j in 1..ny {
            d_hz_y[idx(i, j, ny)] = (hz[idx(i, j, ny)] - hz[idx(i, j - 1, ny)]) / dy;
        }
        d_hz_y[idx(i, 0, ny)] = if per_y != 0 {
            (hz[idx(i, 0, ny)] - hz[idx(i, ny - 1, ny)]) / dy
        } else {
            hz[idx(i, 0, ny)] / dy
        };
    }

    for j in 0..ny {
        for i in 1..nx {
            d_hz_x[idx(i, j, ny)] = (hz[idx(i, j, ny)] - hz[idx(i - 1, j, ny)]) / dx;
        }
        d_hz_x[idx(0, j, ny)] = if per_x != 0 {
            (hz[idx(0, j, ny)] - hz[idx(nx - 1, j, ny)]) / dx
        } else {
            hz[idx(0, j, ny)] / dx
        };
    }
}
