## How to benchmark the reading and streaming-to-cache speed on S3DF

 
### Reserve one or more machines on S3DF to run the tests:

Reserve cores on one on more S3DF nodes like this:
 

    salloc -n 192 --nodes 3 --exclusive --account lcls:prjdat21 -t 12:00:00 -p milano
 
(n is the number of cores, --nodes the number of nodes. The command  spreads the cores evenly across the nodes, which is what we want)

### Start the cache on a Data Movement Node

Run the cache on sdfdtn003

Cache code: 

    code.ornl.gov:99R/nng_stream.git

### Run directly psana_push on S3df using MPI

Runn the streaming code like this:
 
    mpirun -n 64 --map-by ppr:32:node script.sh
 
(map-by pp:32:node will distribute, for example, 32 cores per node)
 
`script.sh` is:
 
    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
    conda deactivate
    conda activate ana-4.0.62-lclstream
    psana_push -e xpptut15 -r 671 -c smd -d epix10k2M -m calib -n 1 -a tcp://134.79.23.43:5000


## Notes:

* Run some scripts to drain the cache and never let it fill up. The precise number of instances of the draining script that should be run depends on how many instances of psana_push are being started
 
* To have an idea of the streaming speed, one can watch the inbound traffic on sdfdtn003 with Grafana: https://grafana.slac.stanford.edu
 
* When using many cores, run 671 of the xpptut experiment ends quite soon. Unfortunately, this is one of the longest runs of this kind that can be found in the tutorial data



Author: Valerio 
