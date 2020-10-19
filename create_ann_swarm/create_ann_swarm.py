import os
os.environ["UW_ENABLE_TIMING"] = "1"

import underworld as uw
import math
from underworld import function as fn
# import glucifer
import numpy as np
# from matplotlib import pyplot as plt
import time
import h5py

order   = int(os.getenv("UW_ORDER",1))
res_env = int(os.getenv("UW_RESOLUTION",32))
itol    = float(os.getenv("UW_SOL_TOLERANCE",1.e-6))
otol    = float(os.getenv("UW_SOL_TOLERANCE",1.e-6))
penalty = float(os.getenv("UW_PENALTY",1e-3))
do_IO   = bool(int(os.getenv("UW_ENABLE_IO","0")))
jobid   = str(os.getenv("PBS_JOBID",os.getenv("SLURM_JOB_ID","0000000")))
ncpus   = int(os.getenv("NTASKS",1))

# In[ ]:

uw.timing.start() #starts timing

# model parameters
res_angular 	= res_env*2
res_radial	= res_env
dim  		= 2
inner_radius    = 3480./6371.
outer_radius 	= 6371./6371.

# model angular extension occurs at 90 degree 
min_angle 	= 62.70
max_angle       = 117.30
# print((max_angle-min_angle)*111)


# In[4]:


# crustal layer thickness in the model
crust_depth = 30
res         = str(res_angular)+'_'+str(res_radial)+'_'+str(ncpus)
# print(res)
outputPath = os.path.join(os.path.abspath("/scratch/n69/tg7098/annulus_swarm/2891_660_annulus/"),"swarm_"+str(res)+"/")
# outputPath = os.path.join(os.path.abspath("."),"swarmMatInd/")
if uw.mpi.rank == 0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.mpi.barrier()


# In[ ]:

mesh                = uw.mesh.FeMesh_Annulus(elementRes=(res_radial, res_angular), 
                                             radialLengths=(inner_radius, outer_radius),
                                             angularExtent=[min_angle, max_angle], 
                                             periodic=[False, False])

velocityField       = mesh.add_variable( nodeDofCount=dim )
pressureField       = mesh.subMesh.add_variable( nodeDofCount=1 )
densityField	    = mesh.add_variable( nodeDofCount=1 )

vc                  = mesh.add_variable( nodeDofCount=dim ) #julesfix
vc_eqNum            = uw.systems.sle.EqNumber( vc, False )  
vcVec               = uw.systems.sle.SolutionVector(vc, vc_eqNum)


# In[ ]:


def rtheta2xy(data):
    """
    converts (r, theta) to (x, y) coordinates
    """
    newcoords 		= np.zeros((len(data[:,0]),2))
    newcoords[:,0] 	= data[:,0]*np.cos(data[:,1]*np.pi/180.0)
    newcoords[:,1] 	= data[:,0]*np.sin(data[:,1]*np.pi/180.0)
    return newcoords


# In[ ]:


def xy2rtheta(data):
    """
    converts (x, y) to (r, theta) coordinates
    """
    newcoords 		= np.zeros((len(data[:,0]),2))
    newcoords[:,0] 	= np.sqrt(data[:,0]**2 + data[:,1]**2)
    newcoords[:,1] 	= np.arctan2(data[:,1],data[:,0]) * 180 / np.pi
    return newcoords


# In[ ]:

# loading shapes of subducting and oceanic plates
input_files_path = '/home/565/tg7098/annulus_input/model_input_660/'
slab_top         = np.genfromtxt(input_files_path+'trans_profile_slab_top.dat',dtype='float')
slab_bottom      = np.genfromtxt(input_files_path+'trans_profile_slab_bottom.dat', dtype='float')
OC_top           = np.genfromtxt(input_files_path+'trans_profile_OC_top.dat', dtype=None)
OC_bottom        = np.genfromtxt(input_files_path+'trans_profile_OC_bottom.dat', dtype='float')
if crust_depth == 10:
    slab_crust = np.genfromtxt(input_files_path+'trans_profile_slab_crust_10.dat',dtype='float')
    OC_crust = np.genfromtxt(input_files_path+'trans_profile_OC_crust_10.dat',dtype='float')
if crust_depth == 20:
    slab_crust = np.genfromtxt(input_files_path+'trans_profile_slab_crust_20.dat',dtype='float')
    OC_crust = np.genfromtxt(input_files_path+'trans_profile_OC_crust_20.dat',dtype='float')
if crust_depth == 30:
    slab_crust = np.genfromtxt(input_files_path+'trans_profile_slab_crust_30.dat',dtype='float')
    OC_crust = np.genfromtxt(input_files_path+'trans_profile_OC_crust_30.dat',dtype='float')
if crust_depth == 40:
    slab_crust = np.genfromtxt(input_files_path+'trans_profile_slab_crust_40.dat',dtype='float')
    OC_crust = np.genfromtxt(input_files_path+'trans_profile_OC_crust_40.dat',dtype='float')
if crust_depth == 50:
    slab_crust = np.genfromtxt(input_files_path+'trans_profile_slab_crust_50.dat',dtype='float')
    OC_crust = np.genfromtxt(input_files_path+'trans_profile_OC_crust_50.dat',dtype='float')


# In[ ]:


# converting (r, thetha) to (x, y) coordinates
slab_top_coords     = rtheta2xy(slab_top[:,[0,1]])
slab_bottom_coords  = rtheta2xy(slab_bottom[:,[0,1]])
OC_top_coords       = rtheta2xy(OC_top[:,[0,1]])
OC_bottom_coords    = rtheta2xy(OC_bottom[:,[0,1]])
slab_crust_coords   = rtheta2xy(slab_crust[:,[0,1]])
OC_crust_coords     = rtheta2xy(OC_crust[:,[0,1]])


# In[ ]:


#creating swarm to model subduction
swarm               = uw.swarm.Swarm(mesh, particleEscape=True)
layout              = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm, particlesPerCell=20)
swarm.populate_using_layout(layout)
materialVariable    = swarm.add_variable( dataType="int",   count=1 )
pol_con             = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=20)


# In[ ]:


# material indices
SlabCrustIndex 	= 1
SlabMantleIndex = 4
OCCrustIndex 	= 2
OCMantleIndex 	= 5
CCrustIndex 	= 3
CMantleIndex 	= 6
UMantleIndex 	= 0
LMantleIndex 	= 7


# In[ ]:


# combining points to form polygon. Note: it should either in clockwise or anti-clockwise order
reversed_slab_crust_coords  = slab_crust_coords[::-1]
reversed_slab_bottom_coords = slab_bottom_coords[::-1]
reversed_OC_bottom_coords   = OC_bottom_coords[::-1]
slabcrust                   = fn.shape.Polygon( np.concatenate((slab_top_coords, reversed_slab_crust_coords), axis=0) )
slabmantle                  = fn.shape.Polygon( np.concatenate((slab_crust_coords, reversed_slab_bottom_coords), axis=0) )
OCMantle                    = fn.shape.Polygon( np.concatenate((OC_crust_coords, reversed_OC_bottom_coords), axis=0) )
OCCrust                     = fn.shape.Polygon( np.concatenate((OC_top_coords, OC_crust_coords), axis=0) )


# In[ ]:


# # Loading swarm and material variable if they exist else create them.
# materialVariable_file = Path(outputPath+"materialVariable.h5")
# if materialVariable_file.is_file():
#     swarm = uw.swarm.Swarm(mesh, particleEscape=True)
#     swarm.load(outputPath+"swarm.h5")
#     materialVariable       = swarm.add_variable("int", 1)
#     materialVariable.load(outputPath+"materialVariable.h5")
#     pol_con = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=20)
# #     pol_con = uw.swarm.PopulationControl(swarm,deleteThreshold=0.025,splitThreshold=0.2)    
# else:
#     # creating swarm and material variable if they don't exist
#     swarm = uw.swarm.Swarm(mesh, particleEscape=True)
#     layout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm, particlesPerCell=20)
#     swarm.populate_using_layout(layout)
#     materialVariable       = swarm.add_variable("int", 1)
#     pol_con = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=20)


# In[ ]:


# indexing the material variable
# if not materialVariable_file.is_file():
materialVariable.data[:] = 0
#indexing lower mantle material
rtheta_coord    = xy2rtheta(swarm.particleCoordinates.data)
indices         = np.argwhere(rtheta_coord[:,0] <= (6371.0-660.0)/6371.0)
materialVariable.data[indices] = LMantleIndex

#indexing missing points in the oceanic crust
indices_slab = np.argwhere(np.logical_and(rtheta_coord[:,0]>=0.99989, rtheta_coord[:,1]>=80.70))
materialVariable.data[indices_slab] = OCCrustIndex

#indexing oceanic plate and slab
for index in range( len(swarm.particleCoordinates.data) ):
    coord = swarm.particleCoordinates.data[index][:]
    if rtheta_coord[:,0][index] > (6371.0-660.0)/6371.0:
        if slabmantle.evaluate(tuple(coord)):
            materialVariable.data[index] = SlabMantleIndex
        if OCMantle.evaluate(tuple(coord)):
            materialVariable.data[index] = OCMantleIndex
        if slabcrust.evaluate(tuple(coord)):
            materialVariable.data[index] = SlabCrustIndex
        if OCCrust.evaluate(tuple(coord)):
            materialVariable.data[index] = OCCrustIndex

#indexing continental crust
tmp_indices_CC1 = np.logical_and(rtheta_coord[:,0] > (6371.0-40.0)/6371.0, rtheta_coord[:,1] < 80.70)
tmp_indices_CC2 = (materialVariable.data[:,0] == 0)
indices_CC      = np.argwhere(np.logical_and(tmp_indices_CC1, tmp_indices_CC2))
materialVariable.data[indices_CC] = CCrustIndex    

#indexing topmost upper mantle of continental plate
tmp_indices_CM1 = np.logical_and(rtheta_coord[:,0] <= (6371.0-40.0)/6371.0, rtheta_coord[:,0] > (6371.0-100.0)/6371.0)
tmp_indices_CM2 = np.logical_and(materialVariable.data[:,0] == 0, rtheta_coord[:,1] < 80.70)
indices_CM      = np.argwhere(np.logical_and(tmp_indices_CM1, tmp_indices_CM2))
materialVariable.data[indices_CM] = CMantleIndex    

# saving swarm and material variable
swarm_copy   = swarm.save(outputPath+'swarm_'+str(res)+'.h5')
mat_Var_copy = materialVariable.save(outputPath+'matVar_'+str(res)+'.h5')


# In[ ]:


# creating mat var xdmf file
materialVariable.xdmf(outputPath+'matVar_'+str(res)+'.xdmf',
                   mat_Var_copy,"materialVariable",swarm_copy,"swarm")


# In[ ]:


# plotting indexed particles
plotting = False
if plotting:
    figParticle = glucifer.Figure(title="Particle Index" )
    figParticle.append( glucifer.objects.Points(swarm, materialVariable, pointSize=2, 
                                                colours='white green red purple blue', discrete=True) )
    lv = figParticle.window()
    lv.rotate('y', 0)
    lv.redisplay()


# In[ ]:


# creating profile points to extract data
profile_pts = False
if profile_pts:
    # profile1
    pts_ang1 = np.linspace(76, 84, 200, endpoint='True')
    pts_depth1 = (6371 - 35)/6371
    pts_rtheta1 = np.zeros((len(pts_ang1), 2))
    pts_rtheta1[:,0] = pts_depth1
    pts_rtheta1[:,1] = pts_ang1
    profile1_pts_xy = rtheta2xy(pts_rtheta1)

    #profile2
    pts_ang2 = np.linspace(73, 81, 200, endpoint='True')
    pts_depth2 = (6371 - 350)/6371
    pts_rtheta2 = np.zeros((len(pts_ang2), 2))
    pts_rtheta2[:,0] = pts_depth2
    pts_rtheta2[:,1] = pts_ang2
    profile2_pts_xy = rtheta2xy(pts_rtheta2)


# In[ ]:


# visualizing the profile points and slab crust point
if profile_pts:
    mesh_profile                = uw.mesh.FeMesh_Annulus(elementRes=(32, 32), 
                                                 radialLengths=(inner_radius, outer_radius),
                                                 angularExtent=[min_angle, max_angle], 
                                                 periodic=[False, False])
    
    slab_bot_top_crust = np.concatenate((slab_top_coords, slab_bot_top, slab_crust_coords), axis=0)
    swarm_slab = uw.swarm.Swarm( mesh=mesh_profile, particleEscape=True )
    swarm_slab_Coords = np.array(slab_bot_top_crust)
    add_slab = swarm_slab.add_particles_with_coordinates(swarm_slab_Coords)
    
    swarm_profile1 = uw.swarm.Swarm( mesh=mesh_profile, particleEscape=True )
    swarm_profile1_Coords = np.array(profile1_pts_xy)
    add_profile1 = swarm_profile1.add_particles_with_coordinates(swarm_profile1_Coords)
    
    swarm_profile2 = uw.swarm.Swarm( mesh=mesh_profile, particleEscape=True )
    swarm_profile2_Coords = np.array(profile2_pts_xy)
    add_profile2 = swarm_profile2.add_particles_with_coordinates(swarm_profile2_Coords)

    fig = glucifer.Figure(figsize=(800,400))
    fig.append( glucifer.objects.Points(swarm=swarm_profile1, pointSize=3, colour='blue', colourBar=False) )
    fig.append( glucifer.objects.Points(swarm=swarm_profile2, pointSize=3, colour='blue', colourBar=False) )
    fig.append( glucifer.objects.Points(swarm=swarm_slab, pointSize=4, colour='red', colourBar=False) )
    fig.append( glucifer.objects.Mesh(mesh_profile))
    lv = fig.window()
    lv.rotate('y', 0)
    lv.redisplay()

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
