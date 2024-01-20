#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#!SBATCH --constraint gpu         # Use GPU 
#SBATCH --partition=milano
#!SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#!SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#!SBATCH --qos=regular        # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time 01:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --job-name=psana_ray
#!SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --nodes=80
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5

cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj.maxie

export USES_MULTI_NODES=true

echo "NODES: $SLURM_JOB_NODELIST"

# Get the list of nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# First node is the head node
head_node=${nodes_array[0]}
echo "Head node: $head_node"

# Extract IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname -I | awk '{print $1}')
echo "Head node IP: $head_node_ip"

# Start Ray on the head node
port=6379
ray start --head --node-ip-address=$head_node_ip --port=$port --num-cpus=$SLURM_CPUS_PER_TASK --block &

# Wait for the head node to initialize
sleep 30

echo "Starting worker nodes..."
# Start Ray on worker nodes
for node in "${nodes_array[@]:1}"; do
    echo "Starting worker node on $node"
    echo "ray start --address=$head_node_ip:$port --num-cpus=$SLURM_CPUS_PER_TASK --block"
    srun --nodes=1 --ntasks=1 -w $node ray start --address=$head_node_ip:$port --num-cpus=$SLURM_CPUS_PER_TASK --block &
done

sleep 30

# Run the Python script
echo "Running Python script..."
## python implement.maxie.psana_loader.ray.py peaknet.cxic00121_0015.yaml
python implement.maxie.psana_loader.ray.py dataloading.xpptut15.yaml

# Shutdown Ray on all nodes
echo "Shutting down nodes..."
for node in "${nodes_array[@]}"; do
    echo "Shutting down Ray on $node"
    srun --nodes=1 --ntasks=1 -w $node ray stop &
done

sleep 30

echo "Job completed"

exit
