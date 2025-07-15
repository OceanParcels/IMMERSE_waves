print('Getting started...')
from glob import glob
import xarray as xr
import numpy as np


###################################################################
### Set path information - needs to be checked before execution ###
print('Set path information...')

datapath = '/storage/shared/oceanparcels/input_data/NEMO4p2_CMCC/'
datapath_nc = datapath + 'MedMFS24_IMMERSE-NEMOv4p2_uncoupled_fulldepth/'
datapath_c = datapath + 'MedMFS24_IMMERSE-NEMOv4p2_coupled_fulldepth/'
gridpath = '/nethome/ruhs0001/DATA/IMMERSE/' 

outpath_data = '/nethome/ruhs0001/IMMERSE_waves/develop-lorenz/data/CalculateSpeed_fulldepth_v5/'

###################################################################


### 1. Open data
print('Opening input velocity data...')

list_ugrid_nc = sorted(glob(datapath_nc + 'MED24_OBC_1d*_grid_U.nc'))
list_vgrid_nc = sorted(glob(datapath_nc + 'MED24_OBC_1d*_grid_V.nc'))
list_ugrid_c = sorted(glob(datapath_c + 'MED24_OBC_1d*_grid_U.nc'))
list_vgrid_c = sorted(glob(datapath_c + 'MED24_OBC_1d*_grid_V.nc'))

# uncoupled experiment contains data from 2018 to 2020, coupled only 2019 to 2020
# -> need to restrict uncoupled one to period of coupled one
uvars_ugrid_nc = xr.open_mfdataset(list_ugrid_nc[365:], combine='by_coords')
vvars_vgrid_nc = xr.open_mfdataset(list_vgrid_nc[365:], combine='by_coords')
uvars_ugrid_c = xr.open_mfdataset(list_ugrid_c, combine='by_coords')
vvars_vgrid_c = xr.open_mfdataset(list_vgrid_c, combine='by_coords')

# mask MedSea (exclude Atlantic part)
mask = xr.open_dataset(outpath_data + 'Mask_MedSea.nc')

# grid for calculating grid sizes (needed for weighted averaging)
grid = xr.open_dataset(gridpath + 'mesh_mask.nc', drop_variables=('x', 'y', 'z'))

print('Performing calculation for each depth level individually...')
for i in range(0,1,1):

    zlev = i + 1
    print(f'Depth level {zlev}:')


    ### 2. Get data on same grid
    print('Interpolating velocities onto same grid...')

    ### 2.1 Rename depth coordinate/dimension (it is the same for both variables)
    uvars_ugrid_depth_nc = uvars_ugrid_nc.isel(depthu=i).rename({'depthu':'z'})
    vvars_vgrid_depth_nc = vvars_vgrid_nc.isel(depthv=i).rename({'depthv':'z'})
    uvars_ugrid_depth_c = uvars_ugrid_c.isel(depthu=i).rename({'depthu':'z'})
    vvars_vgrid_depth_c = vvars_vgrid_c.isel(depthv=i).rename({'depthv':'z'})

    ### 2.2 Interpolate
    vvars_c = vvars_vgrid_depth_c.interp_like(uvars_ugrid_depth_c, method='linear')
    vvars_nc = vvars_vgrid_depth_nc.interp_like(uvars_ugrid_depth_nc, method='linear')
    uvars_c = uvars_ugrid_depth_c
    uvars_nc = uvars_ugrid_depth_nc

    ### 2.3 Mask out Atlantic values
    vvars_c = vvars_c.where(mask.Mask_MedSea == 1)
    vvars_nc = vvars_nc.where(mask.Mask_MedSea == 1)
    uvars_c = uvars_c.where(mask.Mask_MedSea == 1)
    uvars_nc = uvars_nc.where(mask.Mask_MedSea == 1)


    ### 3. Calculate speed
    print('Calculating speed...')

    def calc_velspeed(u,v):
        speed = (u**2 + v**2)**(1/2)
        return speed

    u_Enc = uvars_nc.vozocrtx
    v_Enc = vvars_nc.vomecrty
    speed_Enc = calc_velspeed(u_Enc,v_Enc)

    u_Ec = uvars_c.vozocrtx
    v_Ec = vvars_c.vomecrty
    speed_Ec = calc_velspeed(u_Ec,v_Ec)
        
    u_Sc = uvars_c.usd
    v_Sc = vvars_c.vsd
    speed_Sc = calc_velspeed(u_Sc,v_Sc)

    u_EcSc = u_Ec + u_Sc
    v_EcSc = v_Ec + v_Sc
    speed_EcSc = calc_velspeed(u_EcSc,v_EcSc)

    u_EncSc = u_Enc + u_Sc
    v_EncSc = v_Enc + v_Sc
    speed_EncSc = calc_velspeed(u_EncSc,v_EncSc)

    speed_ScProjEcSc = ((u_EcSc * u_Sc) + (v_EcSc * v_Sc)) / speed_EcSc
    
    speed_EcProjEcSc = ((u_EcSc * u_Ec) + (v_EcSc * v_Ec)) / speed_EcSc


    ### 4. Calculate weighted spatial average of speed
    print('Calculating spatial average...')
    
    def calc_weighted2Dave(tsave):
        zweight_tmp = grid.umask.rename({'nav_lev':'z'}).where(mask.Mask_MedSea == 1)
        zweight_tmp2 = zweight_tmp.isel(z=i).squeeze() * grid.e1u.squeeze() * grid.e2u.squeeze()
        zweight = zweight_tmp2/zweight_tmp2.sum(dim=('x', 'y'), skipna=True)
        xyave = (tsave*zweight).sum(dim=('x', 'y'), skipna=True)
        return xyave

    speed_Enc_rave = calc_weighted2Dave(speed_Enc)
    speed_Ec_rave = calc_weighted2Dave(speed_Ec)
    speed_Sc_rave = calc_weighted2Dave(speed_Sc)
    speed_EcSc_rave = calc_weighted2Dave(speed_EcSc)
    speed_EncSc_rave = calc_weighted2Dave(speed_EncSc)
    speed_ScProjEcSc_rave = calc_weighted2Dave(speed_ScProjEcSc)
    speed_EcProjEcSc_rave = calc_weighted2Dave(speed_EcProjEcSc)
    print('Spatial average calculated.')


    ### 5. Saving data
    print('Saving data ...')

    speed_rave_ds = xr.Dataset(data_vars = {'Enc': speed_Enc_rave.where(speed_Enc_rave !=0),
                                            'Ec': speed_Ec_rave.where(speed_Ec_rave !=0),
                                            'Sc': speed_Sc_rave.where(speed_Sc_rave !=0),
                                            'EcSc': speed_EcSc_rave.where(speed_EcSc_rave !=0),
                                            'EncSc': speed_EncSc_rave.where(speed_EncSc_rave !=0),
                                            'ScProjEcSc': speed_ScProjEcSc_rave.where(speed_ScProjEcSc_rave != 0),
                                            'EcProjEcSc': speed_EcProjEcSc_rave.where(speed_EcProjEcSc_rave != 0)})
    
    if zlev < 10:
        speed_rave_ds.to_netcdf(f'{outpath_data}Speed-rave-weighted-z00{zlev}.nc')
    if (zlev >= 10) and (zlev < 100):
        speed_rave_ds.to_netcdf(f'{outpath_data}Speed-rave-weighted-z0{zlev}.nc')
    if (zlev >= 100):
        speed_rave_ds.to_netcdf(f'{outpath_data}Speed-rave-weighted-z{zlev}.nc')
    print('Spatial average saved.')


print('All done.')
    
