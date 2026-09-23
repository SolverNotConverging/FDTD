"""Prepare, execute, and unpack full compiled Cartesian 2D simulations.

Everything in this module runs outside the numerical time loop. The CPU loop
and CUDA graph replay are native functions in ``_compiled_2d``.
"""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .packing_2d import _compile_tm_events, _compile_te_events, _source_parameters
from .pec_cutcell import pack_enlarged_groups


def compiled_extension():
    try:
        from . import _compiled_2d
    except ImportError as exc:
        raise RuntimeError(
            'The full 2D Cython runtime is unavailable. Run '
            '`python setup_cython.py build_ext --inplace`, or explicitly '
            'select config("python") for reference execution.') from exc
    return _compiled_2d


def select_backend(sim, backend, *, validate=True):
    requested = str(backend).lower().replace('-', '_')
    if requested not in {'cpu', 'gpu', 'python'}:
        raise ValueError("backend must be 'cpu', 'gpu', or 'python'.")
    if validate and requested != 'python':
        extension = compiled_extension()
        if requested == 'gpu':
            try:
                from numba import cuda, config
                if not cuda.is_available():
                    raise RuntimeError('no CUDA device/runtime is available')
                if not config.ENABLE_CUDASIM:
                    extension.check_cuda_driver()
            except Exception as exc:
                raise RuntimeError(f'Full 2D CUDA execution is unavailable: {exc}') from exc
    sim.backend_requested = requested
    sim.backend = {'cpu': 'cython', 'gpu': 'numba_cuda', 'python': 'python'}[requested]
    sim._use_cython_kernel = requested == 'cpu'
    sim._use_numba_cuda = requested == 'gpu'
    return sim


def reference_substep(sim):
    if sim.backend != 'python':
        raise RuntimeError('Individual 2D curl/update methods require config("python"); '
                           'use run() for complete accelerated stepping.')


@dataclass
class Stage:
    name: str
    args: tuple
    # A 2D stage covers the Yee grid, a 1D stage covers events/monitor points.
    work: tuple


def _pack_ade(ade, field, epsilon):
    poles = () if ade is None else ade.poles
    coefficients = np.zeros((len(poles), 3), dtype=np.float64)
    shape = (len(poles),) + field.shape
    r, q, v = (np.zeros(shape, dtype=np.float64) for _ in range(3))
    for index, pole in enumerate(poles):
        coefficients[index] = pole['a'], pole.get('b', 0.0), float('v' in pole)
        r[index] = pole['r']
        q[index] = pole['q']
        if 'v' in pole:
            v[index] = pole['v']
    return dict(coefficients=coefficients, r=r, q=q, v=v,
                sigma=epsilon if ade is None else ade.sigma_term,
                denominator=epsilon if ade is None else ade.denominator,
                original=ade)


def _pack_monitors(sim):
    points, descriptions = [], []
    offset = 0
    for monitor in sim.monitors:
        orientation = monitor.get('orientation', '').lower()
        if orientation not in {'horizontal', 'vertical'}:
            continue
        x0, x1 = int(monitor['ix0']), int(monitor['ix1'])
        y0, y1 = int(monitor['iy0']), int(monitor['iy1'])
        t0, t1 = int(monitor['it0']), int(monitor['it1'])
        length = x1 - x0 if orientation == 'horizontal' else y1 - y0
        duration = max(0, t1 - t0)
        if length <= 0 or duration <= 0:
            continue
        if not (0 <= t0 <= t1 <= sim.Nt):
            raise ValueError('Monitor time windows must lie within [0, Nt].')
        for p in range(length):
            i, j = (x0 + p, y0) if orientation == 'horizontal' else (x0, y0 + p)
            if not (0 <= i < sim.Nx and 0 <= j < sim.Ny):
                raise ValueError('Monitor points must lie inside the cell-centered grid.')
            points.append((i, j, offset + p, length, t0, t1))
        descriptions.append((dict(monitor), offset, duration, length))
        offset += duration * length
    metadata = np.asarray(points, dtype=np.int64).reshape(-1, 6)
    return ([np.ascontiguousarray(metadata[:, i]) for i in range(6)],
            np.empty((offset, 3), dtype=np.float64), descriptions)


def prepare(sim, polarization, record_stride, include_history):
    stride = int(record_stride)
    if stride < 1 or stride != record_stride:
        raise ValueError('record_stride must be a positive integer.')
    if int(sim.Nt) != sim.Nt or sim.Nt < 0:
        raise ValueError('Nt must be a non-negative integer.')
    sim._init_Coeff()
    tm = polarization == 'tm'
    mutable = (('Ez', 'Dz', 'Hx', 'Hy', 'Bx', 'By', 'd_Ez_y', 'd_Ez_x',
                'd_Hx_y', 'd_Hy_x', 'Psi_Bx_y', 'Psi_By_x', 'Psi_Dz_x', 'Psi_Dz_y')
               if tm else
               ('Ex', 'Ey', 'Dx', 'Dy', 'Hz', 'Bz', 'd_Ex_y', 'd_Ey_x',
                'd_Hz_y', 'd_Hz_x', 'Psi_Bz_x', 'Psi_Bz_y', 'Psi_Dx_y', 'Psi_Dy_x'))
    masks = {}
    if tm:
        x_staggered = {'Hx', 'Bx', 'd_Ez_y', 'Psi_Bx_y', 'b_Bx_y', 'c_Bx_y',
                       'kappa_y_Hx', 'CaHx', 'CbHx', 'MRxx_Hx', 'PMC_Hx'}
        y_staggered = {'Hy', 'By', 'd_Ez_x', 'Psi_By_x', 'b_By_x', 'c_By_x',
                       'kappa_x_Hy', 'CaHy', 'CbHy', 'MRyy_Hy', 'PMC_Hy'}
        default_shape = (sim.Nx + 1, sim.Ny + 1)
    else:
        x_staggered = {'Ey', 'Dy', 'd_Hz_x', 'Psi_Dy_x', 'b_Dy_x', 'c_Dy_x',
                       'kappa_x_Ey', 'CaEy', 'CbEy', 'ERyy_Ey', 'PEC_Ey'}
        y_staggered = {'Ex', 'Dx', 'd_Hz_y', 'Psi_Dx_y', 'b_Dx_y', 'c_Dx_y',
                       'kappa_y_Ex', 'CaEx', 'CbEx', 'ERxx_Ex', 'PEC_Ex'}
        default_shape = (sim.Nx, sim.Ny)

    def arrays(*names):
        result = []
        for name in names:
            value = getattr(sim, name)
            shape = ((sim.Nx + 1, sim.Ny) if name in x_staggered else
                     (sim.Nx, sim.Ny + 1) if name in y_staggered else default_shape)
            if value.shape != shape:
                raise ValueError(f'{name} has shape {value.shape}; expected Yee shape {shape}.')
            if name in mutable and not value.flags.writeable:
                raise TypeError(f'{name} must be writable for simulation and synchronization.')
            if name.startswith(('PEC_', 'PMC_')):
                value = masks.setdefault(name, np.ascontiguousarray(value, dtype=np.uint8))
            elif value.dtype != np.float64 or value.ndim != 2 or not value.flags.c_contiguous:
                raise TypeError(f'{name} must be a C-contiguous float64 2D array.')
            result.append(value)
        return tuple(result)

    # Validate public mutable arrays even if a particular stage does not use one.
    arrays(*mutable)
    clock = np.zeros(1, dtype=np.int64)
    grid = (sim.Nx + 1, sim.Ny + 1)
    stages = []

    def add(name, args, work=grid):
        stages.append(Stage(name, tuple(args), work))

    source_parameters = list(_source_parameters(sim.sources))
    for source in sim.sources:
        if source['kind'] not in {'point', 'line-soft', 'sftf', 'waveguide-x', 'waveguide-y'}:
            raise ValueError(f"Unsupported compiled source kind: {source['kind']!r}")
    source_parameters[0] = source_parameters[0].astype(np.int64)
    event_sets = (_compile_tm_events if tm else _compile_te_events)(sim)
    e, h, be, bh, soft = event_sets

    def inject(events, first, second):
        integers = [np.asarray(getattr(events, name), dtype=np.int64)
                    for name in ('target', 'i', 'j', 'source')]
        real = [np.asarray(getattr(events, name), dtype=np.float64)
                for name in ('factor', 'shift')]
        add('_inject_events', (first, second, *integers, *real, events.count,
                              clock, sim.dt, *source_parameters), (events.count,))

    def table(events, first, second):
        indices = [np.asarray(getattr(events, name), dtype=np.int64)
                   for name in ('target', 'i', 'j')]
        values = np.asarray(events.values, dtype=np.float64).reshape(events.count, sim.Nt)
        add('_inject_table_events', (first, second, *indices, values,
                                     events.count, clock), (events.count,))

    dispersive = bool(sim.has_dispersion if tm else sim._has_dispersion())
    ade_states = {}
    for field, epsilon in ((('Ez', 'ERzz_Ez'),) if tm else
                           (('Ex', 'ERxx_Ex'), ('Ey', 'ERyy_Ey'))):
        ade_states[field] = _pack_ade(getattr(sim, '_ade_' + field),
                                      getattr(sim, field), getattr(sim, epsilon))

    def finalize(field, displacement, epsilon, mask):
        ade = ade_states[field]
        add('_finalize_e', (*arrays(field, displacement, epsilon, mask),
                           ade['sigma'], ade['denominator'], ade['coefficients'],
                           ade['r'], ade['q'], ade['v'], sim.dt, dispersive))

    periodic_x = 'x' in getattr(sim, 'periodic', '')
    periodic_y = 'y' in getattr(sim, 'periodic', '')
    cutcell = ()
    if getattr(sim, '_has_cutcell', False):
        if tm:
            cutcell = (sim.pec_open_Hx, sim.pec_open_Hy,
                       *pack_enlarged_groups(sim._pec_enlarged_Hx,
                                             sim.pec_open_Hx, sim.MRxx_Hx),
                       *pack_enlarged_groups(sim._pec_enlarged_Hy,
                                             sim.pec_open_Hy, sim.MRyy_Hy))
        else:
            cutcell = (sim.pec_fluid_area, sim.pec_open_Ex, sim.pec_open_Ey,
                       *pack_enlarged_groups(sim._pec_enlarged_groups,
                                             sim.pec_fluid_area, sim.MRzz_Hz))
    if tm:
        hx_previous, hy_previous = sim.Hx.copy(), sim.Hy.copy()
        add('_tm_curl_e', (*arrays('Ez', 'd_Ez_y', 'd_Ez_x'), sim.dx, sim.dy))
        if cutcell:
            add('_cut_tm_curl_e', (*arrays('Ez', 'd_Ez_y', 'd_Ez_x'),
                                   *cutcell[:2], sim.dx, sim.dy))
        inject(e, sim.d_Ez_y, sim.d_Ez_x)
        table(be, sim.d_Ez_y, sim.d_Ez_x)
        add('_tm_update_h', (*arrays('Hx', 'Hy', 'Bx', 'By'), hx_previous, hy_previous,
            *arrays('d_Ez_y', 'd_Ez_x', 'Psi_Bx_y', 'Psi_By_x',
                    'b_Bx_y', 'c_Bx_y', 'b_By_x', 'c_By_x', 'kappa_y_Hx', 'kappa_x_Hy',
                    'CaHx', 'CbHx', 'CaHy', 'CbHy', 'MRxx_Hx', 'MRyy_Hy', 'PMC_Hx', 'PMC_Hy')))
        if cutcell:
            add('_cut_merge_h', (*arrays('Hx', 'Bx', 'MRxx_Hx'), *cutcell[2:6]),
                (len(cutcell[2]) - 1,))
            add('_cut_merge_h', (*arrays('Hy', 'By', 'MRyy_Hy'), *cutcell[6:10]),
                (len(cutcell[6]) - 1,))
            add('_cut_tm_zero_h', (*arrays('Hx', 'Hy', 'Bx', 'By'), *cutcell[:2]))
        add('_tm_curl_h', (*arrays('Hx', 'Hy', 'd_Hx_y', 'd_Hy_x'), sim.dx, sim.dy,
                           periodic_x, periodic_y))
        inject(h, sim.d_Hx_y, sim.d_Hy_x)
        table(bh, sim.d_Hx_y, sim.d_Hy_x)
        add('_tm_update_e', (*arrays('Ez', 'Dz', 'd_Hx_y', 'd_Hy_x', 'Psi_Dz_x', 'Psi_Dz_y',
            'b_Dz_x', 'c_Dz_x', 'b_Dz_y', 'c_Dz_y', 'kappa_x_Ez', 'kappa_y_Ez',
            'CaEz', 'CbEz', 'ERzz_Ez'), ade_states['Ez']['q'], sim.M, dispersive))
        inject(soft, sim.Dz, sim.Dz)
        finalize('Ez', 'Dz', 'ERzz_Ez', 'PEC_Ez')
    else:
        hz_previous = sim.Hz.copy()
        add('_te_curl_e', (*arrays('Ex', 'Ey', 'd_Ex_y', 'd_Ey_x'), sim.dx, sim.dy))
        if cutcell:
            add('_cut_te_curl_e', (*arrays('Ex', 'Ey', 'd_Ex_y', 'd_Ey_x'),
                                   *cutcell[:3], sim.dx, sim.dy))
        inject(e, sim.d_Ex_y, sim.d_Ey_x)
        table(be, sim.d_Ex_y, sim.d_Ey_x)
        add('_te_update_h', (*arrays('Hz', 'Bz'), hz_previous,
            *arrays('d_Ex_y', 'd_Ey_x', 'Psi_Bz_x', 'Psi_Bz_y', 'b_Bz_x', 'c_Bz_x',
                    'b_Bz_y', 'c_Bz_y', 'kappa_x', 'kappa_y', 'CaHz', 'CbHz', 'MRzz_Hz')))
        if cutcell:
            add('_cut_merge_h', (*arrays('Hz', 'Bz', 'MRzz_Hz'), *cutcell[3:7]),
                (len(cutcell[3]) - 1,))
            add('_cut_te_zero_h', (*arrays('Hz', 'Bz'), cutcell[0]))
        inject(soft, sim.Bz, sim.Bz)
        add('_te_finalize_h', arrays('Hz', 'Bz', 'MRzz_Hz', 'PMC_Hz'))
        add('_te_curl_h', (*arrays('Hz', 'd_Hz_y', 'd_Hz_x'), sim.dx, sim.dy,
                           periodic_x, periodic_y))
        inject(h, sim.d_Hz_y, sim.d_Hz_x)
        table(bh, sim.d_Hz_y, sim.d_Hz_x)
        add('_te_update_e', (*arrays('Ex', 'Ey', 'Dx', 'Dy', 'd_Hz_y', 'd_Hz_x',
            'Psi_Dx_y', 'Psi_Dy_x', 'b_Dx_y', 'c_Dx_y', 'b_Dy_x', 'c_Dy_x',
            'kappa_y_Ex', 'kappa_x_Ey', 'CaEx', 'CbEx', 'CaEy', 'CbEy',
            'ERxx_Ex', 'ERyy_Ey', 'PEC_Ex', 'PEC_Ey'), ade_states['Ex']['q'],
            ade_states['Ey']['q'], sim.M, dispersive))
        finalize('Ex', 'Dx', 'ERxx_Ex', 'PEC_Ex')
        finalize('Ey', 'Dy', 'ERyy_Ey', 'PEC_Ey')

    metadata, monitor_values, descriptions = _pack_monitors(sim)
    monitor_names = ('Ez', 'Hx', 'Hy') if tm else ('Hz', 'Ex', 'Ey')
    previous = (hx_previous, hy_previous) if tm else (hz_previous,)
    # The TE sampler's argument order differs from its output column order.
    monitor_fields = ('Ez', 'Hx', 'Hy') if tm else ('Ex', 'Ey', 'Hz')
    add('_tm_sample_monitors' if tm else '_te_sample_monitors',
        (*arrays(*monitor_fields), *previous, *metadata, monitor_values,
         len(metadata[0]), clock), (len(metadata[0]),))
    record_count = (sim.Nt + stride - 1) // stride if include_history else 0
    history_names = ('Hx', 'Hy', 'Ez', 'Dz') if tm else ('Ex', 'Ey', 'Hz')
    histories = {name: np.empty((record_count,) + getattr(sim, name).shape, dtype=np.float64)
                 for name in history_names}
    add('_tm_record_history' if tm else '_te_record_history',
        (*arrays(*history_names), *histories.values(), clock, stride))
    add('_advance_clock', (clock,), (1,))
    return dict(polarization=polarization, stages=stages, mutable=mutable, ade=ade_states, histories=histories,
                cutcell=cutcell,
                clock=clock, stride=stride, count=record_count, include=bool(include_history),
                monitors=monitor_values, descriptions=descriptions, monitor_names=monitor_names,
                source_events=sum(events.count for events in event_sets),
                monitor_points=len(metadata[0]))


def _finish(sim, prepared):
    for ade in prepared['ade'].values():
        if ade['original'] is not None:
            for index, pole in enumerate(ade['original'].poles):
                pole['q'][...] = ade['q'][index]
                if 'v' in pole:
                    pole['v'][...] = ade['v'][index]
    if any(ade['original'] is not None for ade in prepared['ade'].values()):
        sim._remember_ade_displacement()
    sim.record_stride = prepared['stride']
    sim.is_include_history = prepared['include']
    sim.Nt_rec = prepared['count']
    if prepared['include']:
        for name, values in prepared['histories'].items():
            setattr(sim, name + '_history', values)
    sim.monitor_results = []
    for metadata, offset, duration, length in prepared['descriptions']:
        values = prepared['monitors'][offset:offset + duration * length].reshape(duration, length, 3)
        result = {k: v for k, v in metadata.items() if not k.startswith('_')}
        result.update({name: values[:, :, index].copy()
                       for index, name in enumerate(prepared['monitor_names'])})
        sim.monitor_results.append(result)


class DeviceBuffers:
    """Own device buffers and count real operations at the allocation/copy boundary."""

    def __init__(self, cuda, outputs=()):
        self.cuda = cuda
        self.buffers = {}
        self.operations = Counter()
        self.phase = 'setup'
        self.outputs = {id(value) for value in outputs}

    def upload(self, value):
        if not isinstance(value, np.ndarray):
            return value
        key = id(value)
        if key not in self.buffers:
            device = self.cuda.device_array(value.shape, dtype=value.dtype)
            self.operations[self.phase, 'allocations'] += 1
            if value.size and key not in self.outputs:
                device.copy_to_device(value)
                self.operations[self.phase, 'host_to_device'] += 1
            self.buffers[key] = device
        return self.buffers[key]

    def download(self, value):
        if value.size:
            self.buffers[id(value)].copy_to_host(value)
            self.operations[self.phase, 'device_to_host'] += 1


def _run_gpu(sim, prepared, extension, progress=False):
    from numba import cuda, config
    from . import cuda_2d

    stages = prepared['stages']
    unique = {id(arg): arg for stage in stages for arg in stage.args if isinstance(arg, np.ndarray)}
    required = sum(arg.nbytes for arg in unique.values())
    free, _ = cuda.current_context().get_memory_info()
    # Reserve room for loaded kernels, context overhead and graph instantiation.
    reserve = max(64 * 1024**2, int(required * 0.05))
    if required + reserve > free:
        raise MemoryError(f'2D CUDA run requires {required:,} buffer bytes plus {reserve:,} '
                          f'reserve; {int(free):,} bytes available. Increase record_stride, '
                          'disable histories, or reduce monitor windows/source tables.')
    buffers = DeviceBuffers(cuda, (*prepared['histories'].values(), prepared['monitors']))
    setup_start = perf_counter()
    graph = None
    try:
        # All allocations and uploads precede both graph capture and execution.
        for arg in unique.values():
            buffers.upload(arg)
        launches = []
        for stage in stages:
            if prepared['cutcell'] and stage.name in {'_tm_curl_e', '_te_curl_e'}:
                continue
            if (prepared['polarization'] == 'te' and stage.name == '_finalize_e'
                    and not stage.args[-1]):
                continue  # Nondispersive TE finalization is fused into its E update.
            if stage.work[0] == 0:
                continue
            if 'record_history' in stage.name and not prepared['count']:
                continue
            args = tuple(buffers.upload(arg) for arg in stage.args)
            kernel = getattr(cuda_2d, stage.name)
            # Array columns are contiguous: map CUDA x threads to j so a warp
            # reads adjacent doubles instead of jumping between whole rows.
            threads = cuda_2d.THREADS_2D if len(stage.work) == 2 else cuda_2d.THREADS_1D
            blocks = (((stage.work[1] + threads[0] - 1) // threads[0],
                       (stage.work[0] + threads[1] - 1) // threads[1])
                      if len(stage.work) == 2 else (stage.work[0] + threads - 1) // threads)
            if not config.ENABLE_CUDASIM:
                kernel = kernel.specialize(*args)
            launches.append((kernel, blocks, threads, args))
        if config.ENABLE_CUDASIM:
            # Explicit test harness only: the simulator cannot execute driver graphs.
            # Its measured Python dispatch count intentionally differs from production.
            graph = None
            stream = 0
        else:
            graph = extension.CudaGraph()
            stream = cuda.external_stream(graph.stream)
            graph.begin()
            for kernel, blocks, threads, args in launches:
                kernel[blocks, threads, stream](*args)
                buffers.operations['setup', 'python_dispatch'] += 1
            graph.end()
            if progress:
                graph.prepare_progress(sim.Nt)
        setup_seconds = perf_counter() - setup_start
        buffers.phase = 'steps'
        started = perf_counter()
        if config.ENABLE_CUDASIM:
            from tqdm import tqdm
            for _ in tqdm(range(sim.Nt), desc="FDTD simulation", unit="step",
                          disable=not progress, mininterval=0.2):
                for kernel, blocks, threads, args in launches:
                    kernel[blocks, threads, stream](*args)
                    buffers.operations['steps', 'python_dispatch'] += 1
        else:
            graph.replay(sim.Nt)
        step_seconds = perf_counter() - started
        buffers.phase = 'finish'
        started = perf_counter()
        for name in prepared['mutable']:
            buffers.download(getattr(sim, name))
        for ade in prepared['ade'].values():
            buffers.download(ade['q'])
            buffers.download(ade['v'])
        for history in prepared['histories'].values():
            buffers.download(history)
        buffers.download(prepared['monitors'])
        download_seconds = perf_counter() - started
        stats = {f'{operation}_during_steps': buffers.operations['steps', operation]
                 for operation in ('host_to_device', 'device_to_host', 'allocations', 'python_dispatch')}
        stats.update(source_events=prepared['source_events'], monitor_points=prepared['monitor_points'],
                     buffer_bytes=required, simulator=bool(config.ENABLE_CUDASIM))
        sim._gpu_transfer_stats = stats
        return dict(device_setup_seconds=setup_seconds, stepping_seconds=step_seconds,
                    download_seconds=download_seconds, buffer_bytes=required)
    except Exception as exc:
        # Queued kernels must finish before their buffers are released.
        if graph is not None:
            try:
                graph.synchronize()
            except Exception:
                pass  # Preserve the original launch/copy error; close still runs.
        if buffers.phase == 'setup' and (isinstance(exc, MemoryError)
                or getattr(exc, 'code', None) == 2
                or getattr(exc, 'cuda_error_code', None) == 2):
            raise MemoryError(
                f'Could not allocate the 2D CUDA run ({required:,} buffer bytes; '
                f'{int(free):,} bytes available at preflight). Increase record_stride, '
                'disable histories, or reduce monitor windows/source tables.') from exc
        raise
    finally:
        if graph is not None:
            graph.close()
        buffers.buffers.clear()


def run(sim, polarization, record_stride=1, is_include_history=True, *, progress=False):
    select_backend(sim, sim.backend_requested)
    extension = compiled_extension()
    start = perf_counter()
    prepared = prepare(sim, polarization, record_stride, is_include_history)
    setup_seconds = perf_counter() - start
    if sim.backend == 'cython':
        args = tuple(stage.args for stage in prepared['stages']
                     if not stage.name.startswith('_cut_'))
        start = perf_counter()
        runner = getattr(extension, 'run_' + polarization)
        if prepared['cutcell']:
            runner(args, sim.Nt, sim.Nx, sim.Ny, progress, prepared['cutcell'])
        else:
            runner(args, sim.Nt, sim.Nx, sim.Ny, progress)
        timings = dict(stepping_seconds=perf_counter() - start, download_seconds=0.0)
    else:
        timings = _run_gpu(sim, prepared, extension, progress)
    start = perf_counter()
    _finish(sim, prepared)
    sim._runtime_stats = dict(setup_seconds=setup_seconds,
                              postprocess_seconds=perf_counter() - start, **timings)
