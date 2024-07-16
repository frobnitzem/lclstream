from mpi4py import MPI

from pynng import Push0 # type: ignore[import-untyped]
import zfpy # type: ignore[import-untyped]

from .psana_img_src import PsanaImgSrc

def serialize(data) -> bytes:
    return zfpy.compress_numpy(data, write_header=True)


from pynng import Push0
import click

@click.command
@click.option("--experiment", "-e", help="Experiment identifier", type=str, required=True)
@click.option("--run", "-r", help="Run number", type= int, required=True)
@click.option("--detector", "-d", help="Detector name", type=str, required=True)
@click.option("--mode", "-m", help="Image retrieval mode", type=click.Choice(["raw", "calib", "image", "mask"]), required=True)
@click.option("--addr", "-a", help="Push socket base address", type=str, required=True)
def mpi_img_push(experiment, run, detector, mode, addr):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  ps = PsanaImgSrc(experiment, run, "smd", detector)

  addr_parts = addr.split(":")
  base_curr_addr=":".join(addr_parts[0:-1])
  baseport = int(addr_parts[-1])
  curr_port = baseport + rank
  curr_addr = f"{base_curr_addr}:{curr_port}"
  n = 0
  mbyte = 0
  nbyte = 0
  send_opts = {
     "send_buffer_size": 32 # send blocks if 32 messages queue up
  }
  with Push0(dial=curr_addr, **send_opts) as push:
      print(f"Starting push socket at {curr_addr}")
      for img in ps(mode):
          buf = serialize(img)
          n += 1
          mbyte += img.nbytes
          nbyte += len(buf)
          push.send(buf)
 
if __name__ == "__main__":
    mpi_img_push()
