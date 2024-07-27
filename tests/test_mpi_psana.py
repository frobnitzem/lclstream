from pynng import Pull0 # type: ignore[import-untyped]
import click

@click.command
@click.option("--mpi_rank_size", "-m", help="MPI rank size.", type=int, required=True)
@click.option("--addr", "-a", help="Push socket base address", type=str, required=True)
def run_pull_sockets(mpi_rank_size, addr):
    addr_parts = addr.split(":")
    base_curr_addr=":".join(addr_parts[0:-1])
    baseport = int(addr_parts[-1])

    pull_sockets = []
    for rank in range(mpi_rank_size):
        curr_port = baseport + rank
        curr_addr = f"{base_curr_addr}:{curr_port}"

        pull=Pull0(listen=curr_addr)
        print(f"Starting pull socket at {curr_addr}")
        pull_sockets.append((pull, curr_addr))

    while True:
        for socket_idx,socket in enumerate(pull_sockets):
            data = socket[0].recv()
            print(f"Received message from {socket[1]}")

if __name__ == "__main__":
    run_pull_sockets()
