"""Coordinate source-local dependency builds across distributed nodes."""


def coordinate_node_local_build(
    *, local_rank, compile_dataset, load_fused, barrier
):
    if local_rank == 0:
        compile_dataset()
        load_fused()
    barrier()
    if local_rank != 0:
        load_fused()
    barrier()
