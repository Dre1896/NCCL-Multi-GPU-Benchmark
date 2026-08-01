# NCCL Multi-GPU Communication Benchmark

I ran a real NCCL all-reduce benchmark across 2 rented A100 GPUs to measure how GPU-to-GPU communication actually behaves under a healthy interconnect versus a deliberately degraded one. This project documents the full process: setting up a multi-GPU environment, running a standard NCCL benchmark suite, and proving a real, measurable communication regression with the exact mechanism behind it.

**Result: forcing peer-to-peer communication off dropped average bus bandwidth from 41.86 GB/s to 4.84 GB/s, a roughly 8.6x reduction, caused by NCCL falling back from direct GPU-to-GPU memory access to routing through host memory.**

## Why I built this

Communication cost is one of the least visible bottlenecks in distributed training. A GPU can look fully utilized while the real limiter is how fast it can exchange data with other GPUs, not how fast it can compute. I wanted hands on evidence of what a communication regression actually looks like, not just a description of ring versus tree all-reduce, so I rented real multi-GPU hardware and measured it directly.

## Architecture

```
2x NVIDIA A100 SXM4-80GB (single node, RunPod)
       |
       v
NCCL 2.25.1          — collective communication library, detects GPU topology and selects a communication path
       |
       v
nccl-tests           — official NVIDIA benchmark suite, drives NCCL through a controlled all-reduce workload
       |
       v
NCCL_DEBUG=INFO logs — captures NCCL's actual topology detection and path selection, not just the final numbers
```

I used `all_reduce_perf` specifically because all-reduce is the collective operation most directly tied to distributed training, it is what synchronizes gradients across GPUs after each backward pass.

## Methodology

I provisioned a 2x A100 SXM pod on RunPod, chosen specifically because SXM systems have NVLink between GPUs, which is required to demonstrate a real peer to peer versus fallback comparison. I built `nccl-tests` from source and ran `all_reduce_perf` sweeping message sizes from 8 bytes to 256MB, which exposes both the latency dominated regime at small sizes and the bandwidth dominated regime at large sizes.

I ran the benchmark twice. First as a healthy baseline, with NCCL free to choose its normal communication path. Second with `NCCL_P2P_DISABLE=1` set, which forces NCCL to avoid direct GPU to GPU memory access even though the hardware supports it, simulating a degraded interconnect. I captured full `NCCL_DEBUG=INFO` output for both runs so the actual algorithm and path selection is visible, not just the resulting bandwidth numbers.

## Results

### Summary

| Metric | Healthy (P2P enabled) | Degraded (P2P disabled) |
|---|---|---|
| Communication path | P2P/direct pointer/read | SHM/direct/direct |
| Average bus bandwidth | 41.86 GB/s | 4.84 GB/s |
| Peak bus bandwidth (256MB) | 187.46 GB/s | 13.68 GB/s |
| intraNodeP2pSupport | 1 | 0 |

Bandwidth reduction: approximately 8.6x

### Healthy run, key log lines

```
8a9faed5c3c4:13012:13026 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer/read
```

### Degraded run, key log lines

```
8a9faed5c3c4:13689:13695 [0] NCCL INFO NCCL_P2P_DISABLE set by environment to 1
8a9faed5c3c4:13689:13695 [0] NCCL INFO Check P2P Type intraNodeP2pSupport 0 directMode 1
8a9faed5c3c4:13689:13703 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

### Full bandwidth sweep, healthy run

```
      size (B)     time (us)   busbw (GB/s)
             8         13.26           0.00
          8192         13.45           0.61
        131072         15.44           8.49
       2097152         52.68          39.81
      33554432        232.71         144.19
     268435456       1433.42         187.27

Avg bus bandwidth: 41.7313
```

### Full bandwidth sweep, degraded run

```
      size (B)     time (us)   busbw (GB/s)
             8         14.65           0.00
          8192         17.41           0.47
        131072         62.14           2.11
       2097152        220.31           9.52
      33554432       2466.03          13.61
     268435456      19721.00          13.61

Avg bus bandwidth: 4.83539
```

## Key findings

The healthy run confirms real NVLink usage, not just a theoretical claim. NCCL's own diagnostic output shows `Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer/read`, meaning GPU 0 could read directly from GPU 1's memory without staging through host RAM. Peak bandwidth reached 187.46 GB/s at the largest message size, consistent with real NVLink throughput between two A100s.

Forcing `NCCL_P2P_DISABLE=1` changed the actual communication mechanism, not just its performance. NCCL's log explicitly confirms this with `Check P2P Type intraNodeP2pSupport 0`, direct peer access was disabled, and the path fell back to `SHM/direct/direct`, shared host memory. Instead of one direct GPU to GPU transfer, data now has to copy from GPU 0 into host memory, then from host memory into GPU 1, two extra copies through a slower interconnect. That mechanism, not a vague configuration difference, is what produced the 8.6x bandwidth drop.

The gap is largest at big message sizes, 187.27 GB/s versus 13.61 GB/s at 256MB, because large transfers are bandwidth bound, they saturate whatever the underlying path can sustain. At small message sizes both runs look similar, since small transfers are latency bound and fixed overhead dominates regardless of path. This matches the same latency bound versus bandwidth bound pattern I saw in kernel level profiling on other projects, just applied to inter GPU communication instead of memory access within a single GPU.

## Design notes

I chose to disable P2P rather than attempt artificial network throttling because single node multi GPU systems communicate over NVLink or PCIe, not a real network link, so tools like `tc` for traffic shaping do not meaningfully apply here. Disabling P2P is a legitimate, NCCL native way to force a real fallback path and produce an honest, reproducible comparison.

I used 2 GPUs rather than 4 for the initial comparison to keep the setup simple and the cost low, since the goal was demonstrating the mechanism clearly, not maximizing scale. The same methodology extends directly to more GPUs or more nodes.

## How this extends to larger clusters

At 2 GPUs on a single node, the difference between P2P and shared memory communication is already large, 8.6x here. At real training scale, hundreds or thousands of GPUs across multiple physical machines, the equivalent slow path is not host memory, it is crossing InfiniBand or Ethernet between nodes instead of staying on a fast local fabric. The underlying mechanism is the same: when GPUs cannot use the fastest available path to exchange data, communication time grows and increasingly dominates total step time. This is also the practical content behind Amdahl's law at scale, communication overhead that is negligible at small GPU counts becomes the actual bottleneck once the compute work per GPU shrinks and the communication cost does not shrink with it.

## Getting started

Requirements:
- Access to a multi-GPU instance with NVLink support (A100 SXM, H100 SXM, or similar)
- CUDA toolkit and NCCL installed (most cloud GPU templates include this)

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=0 -j
```

Run the healthy baseline:

```bash
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 2 2>&1 | tee healthy_run.log
```

Run the degraded comparison:

```bash
NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 2 2>&1 | tee degraded_run.log
```

`MPI=0` is used because this test runs on a single node. Multi node testing requires MPI or an equivalent launcher and introduces InfiniBand or Ethernet as the relevant interconnect instead of NVLink versus shared memory.

## Next steps

- Repeat the same comparison at 4 GPUs to see whether the P2P versus fallback gap holds steady or changes with more participants in the communication group
- Extend to true multi node testing to observe InfiniBand behavior directly, rather than the single node NVLink versus PCIe comparison shown here
- Profile the same runs with Nsight Systems to visualize the communication timeline alongside the raw NCCL log data

## License

MIT, see LICENSE.
