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

# In[ ]:

uw.timing.start() #starts timing

# model parameters
res_angular 	= res_env*2
res_radial	    = res_env
dim  		    = 2
inner_radius 	= 3480./6371.
outer_radius 	= 6371./6371.

# model angular extension occurs at 90 degree 
min_angle   = 62.70
max_angle   = 117.30
# print((max_angle-min_angle)*111)

# crustal layer thickness in the model
crust_depth = 30
res 	    = str(res_angular)+'_'+str(res_radial)

# cohesion value and yield stress form
cohesion    = 0.0065
mu          = 0.1


# In[ ]:


#output directory string
'''
tao_Y1: const_coh (C, constant cohesion in the crust)
tao_Y2: coh_mu_rho_g_z (C + mu_rho_g*depth, depth dependent yield stress)
tao_Y3: coh_mu_eff_rho_g_z (C + mu_eff*rho_g*depth, velocity weaking mu)
'''
tao_Y_OC = 'const_coh'


# In[ ]:


# solver options and settings
"""
solver: default (fgmres-mg), mg, mumps, slud (superludist), lu (only serial)
"""
solver 		    = 'fgmres'
inner_rtol 	    = itol   # def = 1e-5
outer_rtol 	    = otol
penalty_mg 	    = penalty
penalty_mumps 	= 1.0e6


# In[ ]:


# adding string to output directory
if tao_Y_OC == 'const_coh':
    file_str 	= str(res)+'_'+str(cohesion)+'_'+str(crust_depth)
if tao_Y_OC == 'coh_mu_rho_g_z':
    file_str 	= str(res)+'_'+str(cohesion)+'_'+str(mu)+'_'+str(crust_depth)
if tao_Y_OC == 'coh_mu_eff_rho_g_z':
    file_str 	= str(res)+'_'+str(cohesion)+'_'+str(mu)+'_'+str(crust_depth)


# In[ ]:


# creating plots
create_plot = False


# In[ ]:


# creating output directory
outputPath = os.path.join(os.path.abspath("/scratch/n69/tg7098/"), "out_"+jobid+"/")
if uw.mpi.rank == 0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.mpi.barrier()


# In[ ]:


# # writing content to a file
# write2file = True
# if uw.mpi.rank == 0:
#     outfile1 = open(outputPath+"pf1_"+str(file_str)+".txt", 'w')
#     outfile2 = open(outputPath+"pf2_"+str(file_str)+".txt", 'w')
#     outfile1.write( "Xcoord, Strainrate, Stress, vx, vy\n")
#     outfile2.write( "Xcoord, Strainrate, Stress, vx, vy\n")
#     outfile1.close()
#     outfile2.close()
# uw.mpi.barrier()


# **Create mesh and finite element variables**

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

#thyagi
strainRateInvField  = mesh.add_variable( nodeDofCount=1 )
vc_mag              = mesh.add_variable( nodeDofCount=1 )


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


# all boundary nodes
outer = mesh.specialSets["MaxI_VertexSet"]
inner = mesh.specialSets["MinI_VertexSet"]
left  = mesh.specialSets["MaxJ_VertexSet"]
right = mesh.specialSets["MinJ_VertexSet"]


# In[ ]:


# loading density field
# densityField.load('../tomo_project/densityField_'+str(res)+'.h5')


# In[ ]:


# timing loading process
if uw.mpi.rank == 0:
    print ('---------------------------------')
    print ("Started loading swarm and matVar")
    print ('---------------------------------')
uw.timing.start()


# In[ ]:


# loading swarm and material variable
swarm_mat_var_path  = '/home/565/tg7098/annulus_input/swarmMatVar/2891_660_annulus/'
swarm 		    = uw.swarm.Swarm(mesh, particleEscape=True)
swarm.load(swarm_mat_var_path+'swarm_'+str(res)+'.h5')
materialVariable    = swarm.add_variable("int", 1)
materialVariable.load(swarm_mat_var_path+'matVar_'+str(res)+'.h5')
pol_con 	    = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=20)
# evalute using swarm variable is set to true
swarm.allow_parallel_nn = True


# In[ ]:


uw.timing.stop()
# printing stats of loading process
if uw.mpi.rank == 0:
    print ('---------------------------------')
    print ("Finished loading swarm and matVar")
    print ('---------------------------------')
uw.timing.print_table()


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


# boundary conditions
# free-slip on all sides; normal component = 0 and tagential component != 0
velBC_FS = uw.conditions.RotatedDirichletCondition( 
                                    variable        = velocityField,
                                    indexSetsPerDof = (outer+inner, left+right),
                                    basis_vectors   = (mesh.bnd_vec_normal, mesh.bnd_vec_tangent))

# No-slip on all sides; normal component = 0 and tagential component = 0
velBC_NS = uw.conditions.RotatedDirichletCondition( 
                                    variable        = velocityField,
                                    indexSetsPerDof = (inner+outer+left+right, inner+outer+left+right),
                                    basis_vectors   = (mesh.bnd_vec_normal, mesh.bnd_vec_tangent))

# outer and inner are free-slip; right and left are no-slip
velBC_FSNS = uw.conditions.RotatedDirichletCondition( 
                                    variable        = velocityField,
                                    indexSetsPerDof = (inner+outer+left+right, left+right),
                                    basis_vectors   = (mesh.bnd_vec_normal, mesh.bnd_vec_tangent))


# **Scaling of parameters**

# In[ ]:


rho_M             = 1.
g_M               = 1.
Height_M          = 1.
viscosity_M       = 1.


# In[ ]:


rho_N             = 80.0  # kg/m**3  note delta rho
g_N               = 9.81    # m/s**2
Height_N          = 1000e3   # m
viscosity_N       = 1e20   #Pa.sec or kg/m.sec


# In[ ]:


#Non-dimensional (scaling)
rho_scaling 		= rho_N/rho_M
viscosity_scaling 	= viscosity_N/viscosity_M
g_scaling 		= g_N/g_M
Height_scaling 		= Height_N/Height_M
pressure_scaling 	= rho_scaling * g_scaling * Height_scaling
time_scaling 		= viscosity_scaling/pressure_scaling
strainrate_scaling 	= 1./time_scaling


# \begin{align}
# {\tau}_N = \frac{{\rho}_{0N}{g}_N{l}_N}{{\rho}_{0M}{g}_M{l}_M} {\tau}_M
# \end{align}
# 
# \begin{align}
# {V}_N = \frac{{\eta}_{0M}}{{\eta}_{0N}}\frac{{\rho}_{0N}{g}_N{{l}_N}^2}{{\rho}_{0M}{g}_M{{l}_M}^2} {V}_M
# \end{align}

# In[ ]:


velocity_scaling = Height_scaling/time_scaling


# In[ ]:


# print(0.015*pressure_scaling/1e6)
# print(0.0065*pressure_scaling/1e6)
# print(0.001*pressure_scaling/1e6)


# In[ ]:


##### viscosity values
upperMantleViscosity 	=  1.0
lowerMantleViscosity 	=  50.0  
slabMantleViscosity     =  1000.0  
slabCrustViscosity      =  1000.0
CCrustViscosity         =  1000.0
CMantleViscosity        =  1000.0

# julesfix - use the pure cartesian velocity instead
strainRate_2ndInvariant = fn.tensor.second_invariant(fn.tensor.symmetric(vc.fn_gradient))


# In[ ]:


# rheology1: viscoplastic crust and rest is newtonian
coord = fn.input()

if tao_Y_OC == 'const_coh':
    cohesion_slab 	= cohesion
    tao_Y_slab 		= cohesion_slab
if tao_Y_OC == 'coh_mu_rho_g_z':
    cohesion_slab 	= cohesion
    mu_rho_g 		= 0.1*1.0*1.  # mu = 0.6, rho = 3300, g = 9.81 
    tao_Y_slab 		= cohesion_slab  + mu_rho_g *(1. - fn.math.sqrt(coord[0]**2. + coord[1]**2.))
if tao_Y_OC == 'coh_mu_eff_rho_g_z':
    cohesion_slab 	= cohesion
    vc_crit 		= (4.4/(velocity_scaling))*(10e-2/(365*24*60*60))   # v_crit = 4.4 cm/yr
    vc_mag 		= fn.math.sqrt(fn.math.dot(vc,vc))
    mu_eff 		= 0.6*(1.-0.7) + 0.6*(0.7/(1.+(vc_mag/vc_crit)))     # mu_s*(1-gamma) + mu_s*(gamma/(1+(v/vc)))
    rho_g 		= 1.0*1.0  # mu = 0.6, rho = 3300, g = 9.81 
    tao_Y_slab 		= cohesion_slab  + mu_eff*rho_g *(1. - fn.math.sqrt(coord[0]**2. + coord[1]**2.))
yielding_slab   = 0.5 * tao_Y_slab / (strainRate_2ndInvariant+1.0e-18)


# In[ ]:


viscoplastic 	= 1.
Non_Newtonian 	= 0.


# In[ ]:


# all are Newtonian except viscoplastic oceanic crust
eta_min = upperMantleViscosity
eta_max = slabCrustViscosity
if viscoplastic:
    slabYieldvisc = fn.exception.SafeMaths(fn.misc.max(eta_min, fn.misc.min(slabCrustViscosity, yielding_slab)))


# In[ ]:


# Non-Newtonian and viscoplastic crust (~30km) and Newtonian mantle
if Non_Newtonian:
    n 			= 3.
    sr_T 		= 1e-4
    creep_dislocation 	= slabMantleViscosity * fn.math.pow(((strainRate_2ndInvariant+1.0e-18)/sr_T), (1.-n)/n)
    creep 		= fn.exception.SafeMaths(fn.misc.min(creep_dislocation,slabMantleViscosity))
    slabYieldvisc       = fn.exception.SafeMaths(fn.misc.min(creep, yielding))


# In[ ]:


# Viscosity function for the materials 
viscosityMap = { UMantleIndex 	: upperMantleViscosity, 
                 LMantleIndex 	: lowerMantleViscosity, 
                 SlabCrustIndex : slabYieldvisc,
                 SlabMantleIndex: slabMantleViscosity,
                 OCCrustIndex   : slabYieldvisc,
                 OCMantleIndex	: slabMantleViscosity,
                 CCrustIndex	: CCrustViscosity,
                 CMantleIndex	: CMantleViscosity}
viscosityFn = fn.branching.map( fn_key = materialVariable, mapping = viscosityMap )


# In[ ]:


# LMDensity = densityField
LMDensity       = 0.0
UMDensity 	= 0.0
slabDensity 	= 1.0
CCrustDensity 	= 0.0
densityMap 	= { UMantleIndex    : UMDensity, 
                    LMantleIndex    : LMDensity, 
                    SlabCrustIndex  : slabDensity,
                    SlabMantleIndex : slabDensity,
                    OCCrustIndex    : slabDensity,
                    OCMantleIndex   : slabDensity,
                    CCrustIndex	    : CCrustDensity,
                    CMantleIndex    : CCrustDensity}
densityFn 	= fn.branching.map( fn_key = materialVariable, mapping = densityMap )


# In[ ]:


buoyancyFn = -1.*densityFn * mesh._fn_unitvec_radial()


# **System Setup**

# In[ ]:


stokesSLE = uw.systems.Stokes(velocityField, 
                              pressureField, 
                              fn_viscosity  = viscosityFn  + 0.*velocityField[0],  
                              #julesfix - ensures the nonlinearity is pickedup
                              voronoi_swarm = swarm,
                              fn_bodyforce  = buoyancyFn, 
                              conditions    = velBC_FS, 
                              _removeBCs    = False )


# In[ ]:


def postSolve():
    #julesfix realign vc using the rotation matrix on stokes
    uw.libUnderworld.Underworld.AXequalsY(
        stokesSLE._rot._cself,
        stokesSLE._velocitySol._cself,
        vcVec._cself,
        False
        )


# In[ ]:


stokesSolver = uw.systems.Solver(stokesSLE)


# In[ ]:


# inner solver type
if solver == 'fmgres':
    pass
if solver == 'lu':
    stokesSolver.set_inner_method("lu")
if solver == 'mumps':
    stokesSolver.set_penalty(penalty_mumps)
    stokesSolver.set_inner_method("mumps")
if solver == 'mg':
    stokesSolver.set_penalty(penalty_mg)
    stokesSolver.set_inner_method("mg")
#     stokesSolver.options.mg.levels = 6
if solver == 'slud':
    stokesSolver.set_inner_method('superludist')


# In[ ]:


# rtol value
if inner_rtol != 'def':
    stokesSolver.set_inner_rtol(inner_rtol)
    stokesSolver.set_outer_rtol(outer_rtol)


# In[ ]:


advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=vc, order=2 ) #julesfix


# **Analysis tools**

# In[ ]:


#The root mean square Velocity
velSquared 	= uw.utils.Integral( fn.math.dot(vc,vc), mesh )
area 		= uw.utils.Integral( 1., mesh )
Vrms 		= math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )


# In[ ]:


if create_plot:
    # plotting indexed particles
    figParticle = glucifer.Figure(title="Particle Index" )
    figParticle.append( glucifer.objects.Points(swarm, materialVariable, pointSize=2, 
                                                colours='white green red purple blue', discrete=True) )

    #Plot of Velocity Magnitude
    figVelocityMag = glucifer.Figure(title="Velocity magnitude"+'_'+str(cohesion) )
    figVelocityMag.append( glucifer.objects.Surface(mesh, fn.math.sqrt(fn.math.dot(vc,vc)), onMesh=True) ) #julesfix

    #Plot of Strain Rate, 2nd Invariant
    figStrainRate = glucifer.Figure(title="Strain rate 2nd invariant"+'_'+str(cohesion), quality=3 )
    figStrainRate.append( glucifer.objects.Surface(mesh, strainRate_2ndInvariant, logScale=True, onMesh=True) )
    figStrainRate.append( glucifer.objects.VectorArrows(mesh, vc, scaling=1, arrowHead=0.2) ) #julesfix

    #Plot of particles viscosity
    figViscosity = glucifer.Figure(title="Viscosity"+'_'+str(cohesion), quality=3 )
    figViscosity.append( glucifer.objects.Points(swarm, viscosityFn, pointSize=2,logScale=True ) )

    #Plot of particles stress invariant
    figStress = glucifer.Figure(title="Stress 2nd invariant"+'_'+str(cohesion), quality=3  )
    figStress.append(glucifer.objects.Points(swarm, 2.0*viscosityFn*strainRate_2ndInvariant, pointSize=2, logScale=True) )


# **Update function**

# In[ ]:


# define an update function
def update():
    # Retrieve the maximum possible timestep for the advection system.
    dt = advector.get_max_dt()
    # Advect using this timestep size.
    advector.integrate(dt)
    return time+dt, step+1


# **Checkpointing function definition**

# In[ ]:


# # variables in checkpoint function
# # Creating viscosity Field
# viscosityField 		= mesh.add_variable(nodeDofCount=1)
# viscVariable          = swarm.add_variable( dataType="float",   count=1 )
# # Creating stress Invariant fn
# stressInvField 		= mesh.add_variable(nodeDofCount=1)
# stressInvFn 		    = 2.0*viscosityField*strainRate_2ndInvariant
# # creating material variable field
# matVarField 		    = mesh.add_variable(nodeDofCount=1)

# variables in checkpoint function
# Creating viscosity Field
viscosityVariable   = swarm.add_variable("double", 1)
stressInvVariable   = swarm.add_variable("double", 1)
viscosityField 	    = mesh.add_variable(nodeDofCount=1)
viscVariable        = swarm.add_variable( dataType="float",   count=1 )
# Creating stress Invariant fn
stressInvFn_swarm   = 2.0*viscosityFn*strainRate_2ndInvariant
stressInvField 	    = mesh.add_variable(nodeDofCount=1)
stressInvFn 	    = 2.0*viscosityField*strainRate_2ndInvariant
# creating material variable field
matVarField 	    = mesh.add_variable(nodeDofCount=1)
# Creating strain rate fn and fields
strainRate 	    = fn.tensor.symmetric(vc.fn_gradient )
strainRateDField    = mesh.add_variable(nodeDofCount=2)
strainRateNDField   = mesh.add_variable(nodeDofCount=2)
# Creating stress fn and fields
stress 		    = 2.*viscosityField*strainRate
stressDField 	    = mesh.add_variable(nodeDofCount=2)
stressNDField 	    = mesh.add_variable(nodeDofCount=2)


# In[ ]:


# meshHnd = mesh.save(   outputPath+'mesh.00000.h5')
# def checkpoint():

#     # save swarm and swarm variables
#     swarmHnd            = swarm.save(           outputPath+'swarm.'           + str(step).zfill(5) +'.h5')
    
#     # Projecting viscosity from particles to viscosity mesh variable
#     matVar_projector 	= uw.utils.MeshVariable_Projection( matVarField, materialVariable, type=0 )
#     matVar_projector.solve()
#     matVarFieldHnd 		= matVarField.save(outputPath+'matVarField.'+ str(step).zfill(5) +'.h5')
    
#     # save mesh variables
#     velocityHnd     = vc.save(      outputPath+'velocityField.'       + str(step).zfill(5) +'.h5', meshHnd) #julesfix
#     pressureHnd     = pressureField.save(      outputPath+'pressureField.'       + str(step).zfill(5) +'.h5', meshHnd)
    
#     # xdmf files
#     matVarField.xdmf(outputPath+'matVarField.'+str(step).zfill(5)+'.xdmf',
#                        matVarFieldHnd,"matVarField",meshHnd,"mesh",modeltime=time)
#     vc.xdmf(       outputPath+'velocityField.'       +str(step).zfill(5)+'.xdmf',
#                        velocityHnd,     "velocity",      meshHnd, "mesh",modeltime=time) #julesfix
#     pressureField.xdmf(       outputPath+'pressureField.'       +str(step).zfill(5)+'.xdmf',
#                        pressureHnd,     "pressure",      meshHnd, "mesh",modeltime=time)

#     # save visualisation
#     if create_plot:
#         figParticle.save(    outputPath + "particle."    + str(step).zfill(5))
#         figVelocityMag.save( outputPath + "velocityMag." + str(step).zfill(5))
#         figStrainRate.save(  outputPath + "strainRate."  + str(step).zfill(5))
#         figViscosity.save(   outputPath + "viscosity."   + str(step).zfill(5))
#         figStress.save(      outputPath + "stress."      + str(step).zfill(5))
        
#     #thyagi
#     strainRateInvField.data[:] 	= strainRate_2ndInvariant.evaluate(mesh)[:]
#     strainRateInvFieldHnd 		=strainRateInvField.save(outputPath+'strainRateInvField.'+str(step).zfill(5)+'.h5', meshHnd)
#     strainRateInvField.xdmf(outputPath+'strainRateInvField.'       +str(step).zfill(5)+'.xdmf',
#                        strainRateInvFieldHnd,     "strainRateInv",      meshHnd, "mesh",modeltime=time)
    
#     viscVariable.data[:] 	= viscosityFn.evaluate(swarm)[:]
#     # Projecting viscosity from particles to viscosity mesh variable
#     visc_projector 			= uw.utils.MeshVariable_Projection( viscosityField, viscVariable, type=0 )
#     visc_projector.solve()

#     viscosityFieldHnd 		= viscosityField.save(outputPath+'viscosityField.'+ str(step).zfill(5) +'.h5')
#     viscosityField.xdmf(outputPath+'viscosityField.'+str(step).zfill(5)+'.xdmf',
#                        viscosityFieldHnd,"viscosityField",meshHnd,"mesh",modeltime=time)
    
#     stressInvField.data[:] 	= stressInvFn.evaluate(mesh)[:]
#     stressInvFieldHnd 		= stressInvField.save(outputPath+'stressInvField.'+ str(step).zfill(5) +'.h5')
#     stressInvField.xdmf(outputPath+'stressInvField.'+str(step).zfill(5)+'.xdmf',
#                        stressInvFieldHnd,"stressInvField",meshHnd,"mesh",modeltime=time)


# In[ ]:


meshHnd = mesh.save(   outputPath+'mesh.00000.h5')
def checkpoint():

    # save swarm and swarm variables
    swarmHnd            = swarm.save(outputPath+'swarm.'+ str(step).zfill(5) +'.h5')
    materialVariableHnd = materialVariable.save(outputPath+'materialVariable.'+ str(step).zfill(5) +'.h5')
    materialVariable.xdmf(outputPath+'materialVariable.'+str(step).zfill(5)+'.xdmf',materialVariableHnd,"materialVariable",swarmHnd,"swarm",modeltime=time)
    
    # Projecting viscosity from particles to viscosity mesh variable
    matVar_projector 	= uw.utils.MeshVariable_Projection( matVarField, materialVariable, type=0 )
    matVar_projector.solve()
    matVarFieldHnd 	= matVarField.save(outputPath+'matVarField.'+ str(step).zfill(5) +'.h5')
    

    # save mesh variables
    velocityHnd     = vc.save(outputPath+'velocityField.'       + str(step).zfill(5) +'.h5', meshHnd) #julesfix
    pressureHnd     = pressureField.save(outputPath+'pressureField.'       + str(step).zfill(5) +'.h5', meshHnd)
    
    # xdmf files
    matVarField.xdmf(outputPath+'matVarField.'+str(step).zfill(5)+'.xdmf', matVarFieldHnd,"matVarField",meshHnd,"mesh",modeltime=time)
    vc.xdmf(outputPath+'velocityField.'       +str(step).zfill(5)+'.xdmf', velocityHnd,     "velocity",      meshHnd, "mesh",modeltime=time) #julesfix
    pressureField.xdmf( outputPath+'pressureField.'       +str(step).zfill(5)+'.xdmf',pressureHnd,     "pressure",      meshHnd, "mesh",modeltime=time)

    # save visualisation
    if create_plot:
        figParticle.save(    outputPath + "particle."    + str(step).zfill(5))
        figVelocityMag.save( outputPath + "velocityMag." + str(step).zfill(5))
        figStrainRate.save(  outputPath + "strainRate."  + str(step).zfill(5))
        figViscosity.save(   outputPath + "viscosity."   + str(step).zfill(5))
        figStress.save(      outputPath + "stress."      + str(step).zfill(5))
        
    #thyagi
    strainRateInvField.data[:] 	= strainRate_2ndInvariant.evaluate(mesh)[:]
    strainRateInvFieldHnd     	= strainRateInvField.save(outputPath+'strainRateInvField.' + str(step).zfill(5) +'.h5',meshHnd)
    strainRateInvField.xdmf(outputPath+'strainRateInvField.'       +str(step).zfill(5)+'.xdmf',strainRateInvFieldHnd,     "strainRateInv",      meshHnd, "mesh",modeltime=time)
    
#     # it is possible to evalute on mesh or swarm
#     viscosityVariable.data[:] = viscosityFn.evaluate(swarm)[:]
#     viscosityVariableHnd = viscosityVariable.save(outputPath+'viscosityVariable.'+ str(step).zfill(5) +'.h5')
#     viscosityVariable.xdmf(outputPath+'viscosityVariable.'+str(step).zfill(5)+'.xdmf',
#                        viscosityVariableHnd,"viscosityVariable",swarmHnd,"swarm",modeltime=time)
    
    stressInvVariable.data[:]   = stressInvFn_swarm.evaluate(swarm)[:]
    stressInvVariableHnd        = stressInvVariable.save(outputPath+'stressInvVariable.'+ str(step).zfill(5) +'.h5')
    stressInvVariable.xdmf(outputPath+'stressInvVariable.'+str(step).zfill(5)+'.xdmf',stressInvVariableHnd,"stressInvVariable",swarmHnd,"swarm",modeltime=time)
    
    viscVariable.data[:] 	= viscosityFn.evaluate(swarm)[:]
    # Projecting viscosity from particles to viscosity mesh variable
    visc_projector 		= uw.utils.MeshVariable_Projection( viscosityField, viscVariable, type=0 )
    visc_projector.solve()

    viscosityFieldHnd 		= viscosityField.save(outputPath+'viscosityField.'+ str(step).zfill(5) +'.h5')
    viscosityField.xdmf(outputPath+'viscosityField.'+str(step).zfill(5)+'.xdmf',viscosityFieldHnd,"viscosityField",meshHnd,"mesh",modeltime=time)
    
    stressInvField.data[:] 	= stressInvFn.evaluate(mesh)[:]
    stressInvFieldHnd 		= stressInvField.save(outputPath+'stressInvField.'+ str(step).zfill(5) +'.h5')
    stressInvField.xdmf(outputPath+'stressInvField.'+str(step).zfill(5)+'.xdmf',stressInvFieldHnd,"stressInvField",meshHnd,"mesh",modeltime=time)
    
    # Assigning strain rate data to fields
    strainRateDField.data[:] 	= strainRate.evaluate(mesh)[:,0:2]
    strainRateNDField.data[:] 	= strainRate.evaluate(mesh)[:,2:4]
    # Creating strain rate h5 and xdmf files
    strainRateDFieldHnd 	= strainRateDField.save(outputPath+'strainRateDField.'+ str(step).zfill(5) +'.h5')
    strainRateDField.xdmf(outputPath+'strainRateDField.'+str(step).zfill(5)+'.xdmf',strainRateDFieldHnd,"strainRateDField",meshHnd,"mesh",modeltime=time)
    strainRateNDFieldHnd 	= strainRateNDField.save(outputPath+'strainRateNDField.'+ str(step).zfill(5) +'.h5')
    strainRateNDField.xdmf(outputPath+'strainRateNDField.'+str(step).zfill(5)+'.xdmf',strainRateNDFieldHnd,"strainRateNDField",meshHnd,"mesh",modeltime=time)
    
    # Assigning strain rate data to fields
    stressDField.data[:] 	= stress.evaluate(mesh)[:,0:2]
    stressNDField.data[:] 	= stress.evaluate(mesh)[:,2:4]
    # Creating strain rate h5 and xdmf files
    stressDFieldHnd 		= stressDField.save(outputPath+'stressDField.'+ str(step).zfill(5) +'.h5')
    stressDField.xdmf(outputPath+'stressDField.'+str(step).zfill(5)+'.xdmf',stressDFieldHnd,"stressDField",meshHnd,"mesh",modeltime=time)
    stressNDFieldHnd 		= stressNDField.save(outputPath+'stressNDField.'+ str(step).zfill(5) +'.h5')
    stressNDField.xdmf(outputPath+'stressNDField.'+str(step).zfill(5)+'.xdmf',stressNDFieldHnd,"stressNDField",meshHnd,"mesh",modeltime=time)


# Main simulation loop
# =======
# 
# The main time stepping loop begins here. Inside the time loop the velocity field is solved for via the Stokes system solver and then the swarm is advected using the advector integrator. Basic statistics are output to screen each timestep.

# In[ ]:


time 		= 0.  # Initial time
step 		= 0   # Initial timestep
maxSteps 	= 2      # Maximum timesteps
steps_output 	= 1   # output every 10 timesteps


# In[ ]:


while step < maxSteps:
    # Solve non linear Stokes system
    stokesSolver.solve(nonLinearIterate=True, callback_post_solve=postSolve, print_stats=True)
    # output figure to file at intervals = steps_output
    if step % steps_output == 0 or step == maxSteps-1:
        pol_con.repopulate()
        checkpoint()
        Vrms = math.sqrt( velSquared.evaluate()[0]/area.evaluate()[0] )
        print ('step = {0:6d}; time = {1:.3e}; Vrms = {2:.3e}'.format(step,time,Vrms))
    # update
    time,step = update()


# In[ ]:


if uw.mpi.rank==0:
    print("Inner (velocity) Solve Options:")
    stokesSolver.options.A11.list()
    print('----------------------------------')
    print("Multigrid (where enabled) Options:")
    stokesSolver.options.mg.list()
    # print("Penalty for mg:", penalty_mg)


# **Post simulation visualisation**

# In[ ]:


if create_plot:
    figParticle.show()
    figVelocityMag.show()
    figStrainRate.show()
    figViscosity.show()
    figStress.show()


# In[ ]:


# creating profile points to extract data
# profile1
pts_ang1 		= np.linspace(76, 84, 200, endpoint='True')
pts_depth1 		= (6371. - 20.)/6371.
pts_rtheta1 		= np.zeros((len(pts_ang1), 2))
pts_rtheta1[:,0] 	= pts_depth1
pts_rtheta1[:,1] 	= pts_ang1
profile1_pts_xy 	= rtheta2xy(pts_rtheta1)

#profile2
pts_ang2 		= np.linspace(75, 83, 200, endpoint='True')
pts_depth2 		= (6371. - 50.)/6371.
pts_rtheta2 		= np.zeros((len(pts_ang2), 2))
pts_rtheta2[:,0] 	= pts_depth2
pts_rtheta2[:,1] 	= pts_ang2
profile2_pts_xy 	= rtheta2xy(pts_rtheta2)


# In[ ]:


# # saving data along the profile1 to txt file
# stress 			= 2.0*viscosityField*strainRate_2ndInvariant
# strainRate_pf1 	= strainRate_2ndInvariant.evaluate_global(profile1_pts_xy)
# vc_pf1 			= vc.evaluate_global(profile1_pts_xy)
# stress_pf1 		= stress.evaluate_global(profile1_pts_xy)

# if write2file:
#     if uw.mpi.rank == 0:
#         outfile = open(outputPath+"pf1_"+str(file_str)+".txt", 'a')
#         for i in zip(profile1_pts_xy[:,0], strainRate_pf1[:,0], stress_pf1[:,0], vc_pf1[:,0], vc_pf1[:,1]):
#             outfile.write("%.6e, %.6e, %.6e, %.6e, %.6e \n" %(i[0],i[1],i[2],i[3],i[4]))
#         outfile.close()


# In[ ]:


# # saving data along the profile2 to txt file
# strainRate_pf2 	= strainRate_2ndInvariant.evaluate_global(profile2_pts_xy)
# vc_pf2 			= vc.evaluate_global(profile2_pts_xy)
# stress_pf2 		= stress.evaluate_global(profile2_pts_xy)

# if write2file:
#     if uw.mpi.rank == 0:
#         outfile = open(outputPath+"pf2_"+str(file_str)+".txt", 'a')
#         for i in zip(profile2_pts_xy[:,0], strainRate_pf2[:,0], stress_pf2[:,0], vc_pf2[:,0], vc_pf2[:,1]):
#             outfile.write("%.6e, %.10e, %.8e, %.10e, %.10e \n" %(i[0],i[1],i[2],i[3],i[4]))
#         outfile.close()


# In[ ]:


# converting r, theta of slab surface into x, y
model_input    = '/home/565/tg7098/annulus_input/model_input_660/'
slab_crust_top = np.genfromtxt(model_input+'trans_profile_slab_top.dat', dtype='float')
slab_crust_15  = np.genfromtxt(model_input+'trans_profile_slab_crust_15.dat', dtype='float')
# slab_crust_30 = np.genfromtxt(model_input+'trans_profile_slab_crust_30.dat', dtype='float')

slab_crust_top_coords = rtheta2xy(slab_crust_top)
slab_crust_15_coords  = rtheta2xy(slab_crust_15)
# slab_crust_30_coords = rtheta2xy(slab_crust_30)


# In[ ]:


# saving data along the interface to txt file using interface coords
coord_list = [slab_crust_top_coords, slab_crust_15_coords, profile1_pts_xy, profile2_pts_xy]
depth_list = ['top', '15', 'profile1', 'profile2']
for i, coords in enumerate(coord_list):
    vc_data                 = velocityField.evaluate_global(coords)
    viscosity_data          = viscosityField.evaluate_global(coords)
    stressInv_data          = stressInvField.evaluate_global(coords)
    strainRateInv_data      = strainRateInvField.evaluate_global(coords)
    strainRateDField_data   = strainRateDField.evaluate_global(coords)
    strainRateNDField_data  = strainRateNDField.evaluate_global(coords)
    stressDField_data	    = stressDField.evaluate_global(coords)
    stressNDField_data      = stressNDField.evaluate_global(coords)
    stressInv_var_data	    = stressInvVariable.evaluate_global(coords)
    if uw.mpi.rank == 0:
        hf_1 = h5py.File(outputPath+str(depth_list[i])+'_data.h5', 'w')
        hf_1.create_dataset('velocity', data=vc_data)
        hf_1.create_dataset('viscosity', data=viscosity_data)
        hf_1.create_dataset('stressInv', data=stressInv_data)
        hf_1.create_dataset('strainRateInv', data=strainRateInv_data)
        hf_1.create_dataset('strainRateDField', data=strainRateDField_data)
        hf_1.create_dataset('strainRateNDField', data=strainRateNDField_data)
        hf_1.create_dataset('stressDField', data=stressDField_data)
        hf_1.create_dataset('stressNDField', data=stressNDField_data)
        hf_1.create_dataset('stressInv_var', data=stressInv_var_data)
        hf_1.close()


# In[ ]:


# #finding the indices of different materials in the model
# SlabCrustIndex	= 1
# SlabMantleIndex	= 4
# OCCrustIndex		= 2
# OCMantleIndex		= 5
# CCrustIndex 		= 3
# CMantleIndex 		= 6
# UMantleIndex 		= 0
# LMantleIndex 		= 7
# SCI 	= (materialVariable.data[:]==1)
# SMI 	= (materialVariable.data[:]==4)
# OCCI 	= (materialVariable.data[:]==2)
# OCMI 	= (materialVariable.data[:]==5)
# CCI 	= (materialVariable.data[:]==3)
# CMI 	= (materialVariable.data[:]==6)
# UMI 	= (materialVariable.data[:]==0)
# LMI 	= (materialVariable.data[:]==7)


# In[ ]:


# #plotting stress vs strain rate for different materials
# fig = plt.figure()
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[SCI]*strainRate_2ndInvariant.evaluate(swarm)[SCI],\
#            strainRate_2ndInvariant.evaluate(swarm)[SCI],'C1.', label='SC')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[OCCI]*strainRate_2ndInvariant.evaluate(swarm)[OCCI],\
#            strainRate_2ndInvariant.evaluate(swarm)[OCCI],'C2.', label='OCC')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[CCI]*strainRate_2ndInvariant.evaluate(swarm)[CCI],\
#            strainRate_2ndInvariant.evaluate(swarm)[CCI],'C3.', label='CC')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[SMI]*strainRate_2ndInvariant.evaluate(swarm)[SMI],\
#            strainRate_2ndInvariant.evaluate(swarm)[SMI],'C4.', label='SM')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[OCMI]*strainRate_2ndInvariant.evaluate(swarm)[OCMI],\
#            strainRate_2ndInvariant.evaluate(swarm)[OCMI],'C5.', label='OCM')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[CMI]*strainRate_2ndInvariant.evaluate(swarm)[CMI],\
#            strainRate_2ndInvariant.evaluate(swarm)[CMI],'C6.', label='CM')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[LMI]*strainRate_2ndInvariant.evaluate(swarm)[LMI],\
#            strainRate_2ndInvariant.evaluate(swarm)[LMI],'C7.', label='LM')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[UMI]*strainRate_2ndInvariant.evaluate(swarm)[UMI],\
#            strainRate_2ndInvariant.evaluate(swarm)[UMI],'C0.', label='UM')
# plt.xlabel('Deviatoric stress')
# plt.ylabel('strain rate')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.show()
# fig.savefig(outputPath+'rheology.png', dpi=300)


# In[ ]:


# fig = plt.figure()
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[SCI]*strainRate_2ndInvariant.evaluate(swarm)[SCI],\
#            strainRate_2ndInvariant.evaluate(swarm)[SCI],'C1.', label='SC')
# plt.loglog(2.0*viscosityFn.evaluate(swarm)[OCCI]*strainRate_2ndInvariant.evaluate(swarm)[OCCI],\
#            strainRate_2ndInvariant.evaluate(swarm)[OCCI],'C2.', label='OCC')
# # plt.loglog(2.0*viscosityFn.evaluate(swarm)[CCI]*strainRate_2ndInvariant.evaluate(swarm)[CCI],\
# #            strainRate_2ndInvariant.evaluate(swarm)[CCI],'C3.', label='CC')
# # # plt.loglog(2.0*viscosityFn.evaluate(swarm)[SMI]*strainRate_2ndInvariant.evaluate(swarm)[SMI],\
# # #            strainRate_2ndInvariant.evaluate(swarm)[SMI],'C4.', label='SM')
# # # plt.loglog(2.0*viscosityFn.evaluate(swarm)[OCMI]*strainRate_2ndInvariant.evaluate(swarm)[OCMI],\
# # #            strainRate_2ndInvariant.evaluate(swarm)[OCMI],'C5.', label='OCM')
# # plt.loglog(2.0*viscosityFn.evaluate(swarm)[CMI]*strainRate_2ndInvariant.evaluate(swarm)[CMI],\
# #            strainRate_2ndInvariant.evaluate(swarm)[CMI],'C6.', label='CM')
# # # plt.loglog(2.0*viscosityFn.evaluate(swarm)[LMI]*strainRate_2ndInvariant.evaluate(swarm)[LMI],\
# # #            strainRate_2ndInvariant.evaluate(swarm)[LMI],'C7.', label='LM')
# # # plt.loglog(2.0*viscosityFn.evaluate(swarm)[UMI]*strainRate_2ndInvariant.evaluate(swarm)[UMI],\
# # #            strainRate_2ndInvariant.evaluate(swarm)[UMI],'C0.', label='UM')
# plt.xlabel('Deviatoric stress')
# plt.ylabel('strain rate')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.show()
# fig.savefig(outputPath+'rheology1.png', dpi=300)


# In[ ]:


# performing velocity scaling 
# mm_yr = 1e3*365*24*60*60
# np.sqrt(vc_pf1[:,0]**2 + vc_pf1[:,1]**2)*velocity_scaling*mm_yr

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
