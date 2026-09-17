# Profiling Metrics — what we collect and what it tells us

Three kernels, two MPI strategies. Profiling has to support comparison across
kernels (which sparsity pattern wins where), and across MPI strategies
(does halo exchange beat gather?).

## Nsight Compute (`ncu`) — per-kernel microanalysis

Run with `ncu --set full --import-source yes -o <out>.ncu-rep ./bin ...`.
For every kernel at its headline workload (N=8192, d=64, H=16, FP32, plus
w=256 for windowed and ρ=0.1 for block-sparse), record into
`report/data/ncu_metrics.csv`:

| Metric ID | Reading | Why it matters |
|-----------|---------|----------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM throughput (%) | Compute-bound detector. |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Memory throughput (%) | Memory-bound detector. |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy (%) | Latency-hiding capacity. |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second` | Global load throughput | Coalescing check (critical for block-sparse — CSR indirection can scatter loads). |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | SMEM bank conflicts | Tile padding health. |
| `launch__registers_per_thread` | Registers/thread | Register pressure vs. occupancy. |
| `smsp__inst_executed.sum` + `sass_thread_inst_executed_op_fadd_…` | FLOPs | Roofline x-axis numerator. |
| `dram__bytes.sum` | HBM bytes moved | Roofline x-axis denominator. |
| `smsp__warp_issue_stalled_*` (all reasons) | Stall reason mix | Which ceiling we actually hit. |
| `gpc__cycles_elapsed.max - gpc__cycles_elapsed.min` (across blocks) | **Block-time variance** | **Block-sparse load imbalance** — directly measures the unevenness across CUDA blocks. |

The last metric matters specifically for block-sparse: each CUDA block
processes one query block-row, and per-block work scales with that row's nnz.
A wide block-time distribution means tail blocks dominate the kernel time.

## Nsight Systems (`nsys`) — end-to-end timeline

Run with `nsys profile -t cuda,nvtx,mpi -o <out> mpirun -np P ./mpi-bin ...`
for every MPI run. We have two MPI strategies, so we want one timeline per
strategy:

1. **Head-parallel.** Are `MPI_Bcast` (Q/K/V) and `MPI_Gather` (O) overlapping
   with kernel work once we move from blocking to per-head Isend/Irecv?
2. **Sequence-parallel windowed.** Is the halo `Isend/Irecv` posted *before*
   the kernel launches, so it overlaps the interior compute? (If not, we are
   wasting the entire point of sequence-parallel.)

Dump a one-line summary per run into `report/data/nsys_summary.csv`:
`run_id, mpi_strategy, total_ms, kernel_ms, mpi_bcast_ms, mpi_gather_ms,
mpi_halo_ms, h2d_ms, d2h_ms, idle_ms`.

## Measured arithmetic intensity (per kernel)
For every variant, compute:
```
AI_measured = FLOPs_executed / dram__bytes_total
```
and plot on the Roofline against the analytical bounds in
`analysis/roofline/machine_peaks.md`:
- Dense: expect AI ≈ N/B (high; compute-bound at large N).
- Windowed: expect AI ≈ 2w/B (flat in N; compute-bound but lower than dense).
- Block-sparse: expect AI ≈ ρN/B in the ideal case; **the gap to this bound
  is the cost of CSR indirection** and is itself a finding.

## Approximation-quality analysis (sparse kernels only)

For windowed and block-sparse, on each synthetic input (`mostly_local`,
`global_needles`, `uniform`) and each (w, ρ) sweep point, record:
- `l2_error  = ||O_kernel - O_dense||_2 / ||O_dense||_2`
- `linf_error = max_{ij} |O_kernel - O_dense|`

This is what tells us when sparsity is fast *and* good. Lives next to the
performance CSVs in `report/data/approx_quality.csv`.

## Archive convention
Every `.ncu-rep` and `.nsys-rep` filename:
`{kernel}_{N}_{d}_{H}_{w_or_rho}_{commit7}.{ncu|nsys}-rep`.
Raw reports go in `analysis/profiling/runs/` (gitignored binaries belong in
LFS or out-of-repo storage; commit only the CSV summaries).
