#[inline]
fn idx(i: usize, j: usize, ny: usize) -> usize {
    i * ny + j
}

#[no_mangle]
pub unsafe extern "C" fn calculate_curl_e_ez(
    ez: *const f64,
    d_ez_x: *mut f64,
    d_ez_y: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    per_x: i32,
    per_y: i32,
) {
    let ez = std::slice::from_raw_parts(ez, nx * ny);
    let d_ez_x = std::slice::from_raw_parts_mut(d_ez_x, nx * ny);
    let d_ez_y = std::slice::from_raw_parts_mut(d_ez_y, nx * ny);

    for i in 0..nx {
        for j in 0..(ny - 1) {
            d_ez_y[idx(i, j, ny)] = (ez[idx(i, j + 1, ny)] - ez[idx(i, j, ny)]) / dy;
        }
        d_ez_y[idx(i, ny - 1, ny)] = if per_y != 0 {
            (ez[idx(i, 0, ny)] - ez[idx(i, ny - 1, ny)]) / dy
        } else {
            -ez[idx(i, ny - 1, ny)] / dy
        };
    }

    for j in 0..ny {
        for i in 0..(nx - 1) {
            d_ez_x[idx(i, j, ny)] = (ez[idx(i + 1, j, ny)] - ez[idx(i, j, ny)]) / dx;
        }
        d_ez_x[idx(nx - 1, j, ny)] = if per_x != 0 {
            (ez[idx(0, j, ny)] - ez[idx(nx - 1, j, ny)]) / dx
        } else {
            -ez[idx(nx - 1, j, ny)] / dx
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn calculate_curl_h_ez(
    hx: *const f64,
    hy: *const f64,
    d_hx_y: *mut f64,
    d_hy_x: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    per_x: i32,
    per_y: i32,
) {
    let hx = std::slice::from_raw_parts(hx, nx * ny);
    let hy = std::slice::from_raw_parts(hy, nx * ny);
    let d_hx_y = std::slice::from_raw_parts_mut(d_hx_y, nx * ny);
    let d_hy_x = std::slice::from_raw_parts_mut(d_hy_x, nx * ny);

    for i in 0..nx {
        for j in 1..ny {
            d_hx_y[idx(i, j, ny)] = (hx[idx(i, j, ny)] - hx[idx(i, j - 1, ny)]) / dy;
        }
        d_hx_y[idx(i, 0, ny)] = if per_y != 0 {
            (hx[idx(i, 0, ny)] - hx[idx(i, ny - 1, ny)]) / dy
        } else {
            hx[idx(i, 0, ny)] / dy
        };
    }

    for j in 0..ny {
        for i in 1..nx {
            d_hy_x[idx(i, j, ny)] = (hy[idx(i, j, ny)] - hy[idx(i - 1, j, ny)]) / dx;
        }
        d_hy_x[idx(0, j, ny)] = if per_x != 0 {
            (hy[idx(0, j, ny)] - hy[idx(nx - 1, j, ny)]) / dx
        } else {
            hy[idx(0, j, ny)] / dx
        };
    }
}
