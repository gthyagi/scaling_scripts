import os
os.environ["UW_ENABLE_TIMING"] = "1"
import underworld as uw
import numpy as np
from underworld import function as fn
import math

# **Model Resolution**

# In[2]:


order   = int(os.getenv("UW_ORDER",1))
res_rad = int(os.getenv("UW_RES_RAD",32))
res_lon = int(os.getenv("UW_RES_LON",48))
res_lat = int(os.getenv("UW_RES_LAT",64))
itol    = float(os.getenv("UW_SOL_TOLERANCE",1.e-6))
otol    = float(os.getenv("UW_SOL_TOLERANCE",1.e-6))
penalty = float(os.getenv("UW_PENALTY",1e-3))
do_IO   = bool(int(os.getenv("UW_ENABLE_IO","0")))
jobid   = str(os.getenv("PBS_JOBID",os.getenv("SLURM_JOB_ID","0000000")))
ncpus   = int(os.getenv("NTASKS",1))

uw.timing.start() #starts timing

resX = res_lon 
resY = res_lat
resZ = res_rad
res  = str(resZ)+str(resX)+str(resY)


# In[3]:


outputPath = os.path.join(os.path.abspath("/scratch/n69/tg7098/spherical_swarm/"),"swarm_"+str(res)+"_"+str(ncpus)+"/")
# outputPath = os.path.join(os.path.abspath("."),"swarmMatInd/")
if uw.mpi.rank == 0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.mpi.barrier()


# **Model extent info in geographical coordinates**

# In[4]:


# max, min values of lon & lat in the region
lon_max 	= 122.0
lon_min 	= 57.0
lat_max 	= 35.0
lat_min 	= -52.0
diff_lon 	= lon_max - lon_min
diff_lat 	= lat_max - lat_min
mid_lon 	= (lon_max + lon_min)/2
mid_lat 	= (lat_max + lat_min)/2
# print (diff_lon, diff_lat)


# **Model extent info in spherical coordinates (i.e., model domain)**

# In[5]:


# transforming lon, lat to spherical range
if lon_min <= 0:
    trans_lon_min = 360 + lon_min
else:
    trans_lon_min = lon_min
if lon_max <= 0:
    trans_lon_max = 360 + lon_max
else:
    trans_lon_max = lon_max
    
if lat_min < 0:
    trans_lat_min = 90 - lat_min
else:
    trans_lat_min = 90 - lat_min
if lat_max < 0:
    trans_lat_max = 90 - lat_max
else:
    trans_lat_max = 90 - lat_max
# print (trans_lon_min, trans_lon_max, trans_lat_min, trans_lat_max)


# **Mesh information**

# In[6]:


"""
mesh information
nodes in each direction (8,12,12) = (radial, long, lat)
min number of nodes in lon lat direction such that no particle is eject from the spherical domain
"""
mesh = uw.mesh.FeMesh_SRegion(elementRes        =(resZ,resX,resY), 
                                  radialLengths =(3480./6371.,6371./6371.),
                                  latExtent     =diff_lat,
                                  longExtent    =diff_lon)


# **Creating swarm**

# In[7]:


swarm 		= uw.swarm.Swarm( mesh=mesh, particleEscape=True )
swarmLayout     = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
swarm.populate_using_layout( layout=swarmLayout )


# In[8]:


materialVariable       		= swarm.add_variable( dataType="int",   count=1 )
materialVariable.data[:] 	= 0


# **Saving swarm, materialIndex in .h5**

# In[9]:


swarm.save(outputPath + 'swarm_'+str(res)+'_'+str(ncpus)+'.h5')
materialVariable.save(outputPath + 'matVar_'+str(res)+'_'+str(ncpus)+'.h5')

uw.timing.stop()
module_timing_data_orig = uw.timing.get_data(group_by="routine")

# write out data
filename = "Res_{}_Nproc_{}_JobID_{}".format(res,uw.mpi.size,jobid)
import json
if module_timing_data_orig:
    module_timing_data = {}
    for key,val in module_timing_data_orig.items():
        module_timing_data[key[0]] = val
    module_timing_data["Other_data"]   = { "res":res, "nproc":uw.mpi.size }
    with open(filename+".json", 'w') as fp:
        json.dump(module_timing_data, fp,sort_keys=True, indent=4)

uw.timing.print_table(group_by="routine", output_file=filename+".txt", display_fraction=0.99)