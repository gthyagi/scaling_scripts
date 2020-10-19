import h5py
import sys

new_dir_name = sys.argv[1] # reading arguments from the terminal
old_dir_name = sys.argv[2]
new_scratch = '/scratch/n69/tg7098/spherical_swarm/'
old_scratch = '/scratch/n69/tg7098/weak_scaling_swarm/'
res = new_dir_name.split("_")[1]
ncpus = new_dir_name.split("_")[2]

old_matVar = h5py.File(old_scratch+old_dir_name+'/matVar_'+res+'_cor_ipb.h5', 'r') # reading mat var h5 files
new_matVar = h5py.File(new_scratch+new_dir_name+'/matVar_'+res+'_'+ncpus+'.h5', 'a') # write mat var h5 files

old_matVar_data = old_matVar['data']
new_matVar_data = new_matVar['data']

# writing new file
file_size   = new_matVar_data.shape[0]
chunk_size  = 1000000
accum_chunk_size = 0  # accumulated chunk size
while accum_chunk_size <= file_size:
    chunk_start = accum_chunk_size
    chunk_end = accum_chunk_size + chunk_size
    if chunk_end > file_size:
        chunk_end = file_size
    print('Start: ', chunk_start, ' End: ', chunk_end)
    old_data_chunk = old_matVar_data[chunk_start:chunk_end]
    new_matVar_data[chunk_start:chunk_end] = old_data_chunk
    accum_chunk_size += chunk_size # updating accumulated chunk size

old_matVar.close()
new_matVar.close()
