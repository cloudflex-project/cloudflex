import numpy as np
from unyt import cm, K, proton_mass
import os
import glob
from IPython import embed

from cloudflex import \
    SpectrumGenerator, \
    plot_histogram, \
    plot_clouds, \
    Clouds, \
    Rays, \
    create_and_plot_clouds, \
    plot_covering_frac, \
    plot_clouds_buffer, \
    make_quad_plot
    

def makeall_clouds(flg=1):

    ## Flags
    # flg=1 -- vary mclmin
    
    # Store everything in parameters dictionary to pass to Clouds class
    params = {}

    # Turbulence Generation Stuff
    N = 128                             # Default: 128
    seed = int(2) # random seed
    np.random.seed(seed=seed)
    dtype = np.float64 # data precision

    params['seed'] = seed
    params['N'] = N
    params['n'] = [N, N, N]
    params['kmin'] = int(1)             # Default: 1
    params['kmax'] = int(N/2)           # Default: N/2
    params['f_solenoidal'] = 2/3.       # Default: 2/3
    params['beta'] = 1/3.               # Default: 1/3
    params['vmax'] = 30                 # Default: 30 km/s

    #### Cloud Generation Stuff
    params['alpha'] = 2                 # Default: 2
    params['zeta'] = -1                 # Default: -1

    # cloud mass min/max (Msun)
    params['mclmin'] = 1e1              # Default: 10 Msun
    params['mclmax'] = 1e5              # Default: 100000 Msun

    # cloud position from center min/max (kpc)
    params['dclmin'] = 0.1              # Default: 0.1 kpc
    params['dclmax'] = 5                # Default: 5 kpc
    params['center'] = [0.0, 0.0]       # Default: (0, 0)

    # background density and temperature of hot medium
    # density and temperature of cool clouds
    params['n_bg'] = 1e-4 / cm**3       # Default: 1e-4 cm^-3
    params['T_bg'] = 1e6 * K            # Default: 1e6 K
    params['P'] = params['n_bg'] * params['T_bg']
    params['T_cl'] = 1e4 * K            # Default: 1e4 K
    params['n_cl'] = params['P'] / params['T_cl']  # Default: 1e-2 cm^-3
    params['Z_cl'] = 0.33               # Default: 0.33
    params['rho_cl'] = params['n_cl'] * proton_mass
    params['total_mass'] = 1e6          # Default: 1e6 Msun
    params['clobber'] = True            # Default: True

    # Ensuring seed state is same for each iteration
    state = np.random.get_state()

    if (flg==1):

        ## Mcl,min loop
        lgmclmin_low = -1
        lgmclmin_high = 5
        #mclmins = np.logspace(lgmclmin_low, lgmclmin_high, num=4)
        mclmins = np.array([0.001])
        
        print("Looping over mclmins = %s" % mclmins)

        uppath = './MCLMIN'
        
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(mclmins)):
            np.random.set_state(state)
            mclmin = mclmins[i]
            print("Generating clouds for mclmin %.1g" % mclmin)
            if mclmin < 1:
                path = './%.1g' % mclmin
                cloud_fil = 'clouds_mclmin_%.1g.h5' % mclmin
            else:
                path = './%1d' % mclmin
                cloud_fil = 'clouds_mclmin_%1d.h5' % mclmin
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['mclmin'] = mclmin
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')


    elif (flg==2):

        ## ncl loop
        lgncl_low = -3
        lgncl_high = -1
        ncls = np.logspace(lgncl_low, lgncl_high, num=5)

        print("Looping over ncls = %s" % ncls)

        uppath = './NCL'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(ncls)):
            np.random.set_state(state)
            ncl = ncls[i]
            print("Generating clouds for ncl %.1g" % ncl)
            if ncl < 1:
                path = './%.1g' % ncl
                cloud_fil = 'clouds_ncl_%.1g.h5' % ncl
            else:
                path = './%1d' % ncl
                cloud_fil = 'clouds_ncl_%1d.h5' % ncl
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            
            params['n_cl'] = ncl / cm**3       # Default: 1e-2 cm^-3
            params['rho_cl'] = params['n_cl'] * proton_mass
            params['P'] = params['T_cl'] * params['n_cl']
            params['n_bg'] = params['P']/params['T_bg']
            
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')


        
    elif (flg==3):

        ## mtot loop
        lgmtot_low = 5
        lgmtot_high = 7
        mtots = np.logspace(lgmtot_low, lgmtot_high, num=5)

        print("Looping over mtots = %s" % mtots)

        uppath = './MTOT'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(mtots)):
            np.random.set_state(state)
            mtot = mtots[i]
            print("Generating clouds for mtot %.1g" % mtot)
            path = './%.1g' % mtot
            cloud_fil = 'clouds_mtot_%.1g.h5' % mtot
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['total_mass'] = mtot
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')



    elif (flg==4):

        ## mtot loop
        #lgdclmax_low = 0.3
        #lgdclmax_high = 1.05
        dclmaxs = np.array([2.2,3.3,5.0,7.5,11.3])

        print("Looping over dclmaxs = %s" % dclmaxs)

        uppath = './DCLMAX'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(dclmaxs)):
            np.random.set_state(state)
            dclmax = dclmaxs[i]
            print("Generating clouds for dclmax %.3g" % dclmax)
            path = './%.3g' % dclmax
            cloud_fil = 'clouds_dclmax_%.3g.h5' % dclmax
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['dclmax'] = dclmax
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')


    elif (flg==5):

        ## ncl loop
        lgZcl_low = -1.5
        lgZcl_high = 0.5
        Zcls = np.logspace(lgZcl_low, lgZcl_high, num=5)

        print("Looping over Zcls = %s" % Zcls)

        uppath = './ZCL'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(Zcls)):
            np.random.set_state(state)
            Zcl = Zcls[i]
            print("Generating clouds for Zcl %.1g" % Zcl)
            if Zcl < 1:
                path = './%.1g' % Zcl
                cloud_fil = 'clouds_Zcl_%.1g.h5' % Zcl
            else:
                path = './%1d' % Zcl
                cloud_fil = 'clouds_Zcl_%1d.h5' % Zcl
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['Z_cl'] = Zcl
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')


    elif (flg==6):

        ## vmax loop
        vmaxs = np.array([8.0, 20.0, 30.0, 60.0, 100.0])

        print("Looping over vmaxs = %s" % vmaxs)

        uppath = './VMAX'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(vmaxs)):
            np.random.set_state(state)
            vmax = vmaxs[i]
            print("Generating clouds for vmax %.1g" % vmax)
            path = './%.1g' % vmax
            cloud_fil = 'clouds_vmax_%.1g.h5' % vmax
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['vmax'] = vmax
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')



    elif (flg==7):

        ## beta loop
        betas = np.array([0.0, 0.2, 0.33, 0.5, 1.0])

        print("Looping over betas = %s" % betas)

        uppath = './BETA'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(betas)):
            np.random.set_state(state)
            beta = betas[i]
            print("Generating clouds for beta %.1g" % beta)
            if beta < 1:
                path = './%.1g' % beta
                cloud_fil = 'clouds_beta_%.1g.h5' % beta
            else:
                path = './%1d' % beta
                cloud_fil = 'clouds_beta_%1d.h5' % beta
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['beta'] = beta
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')


    elif (flg==8):

        ## alpha loop
        alphas = np.array([1.1, 1.5, 2.0, 2.5, 4.0])

        print("Looping over alphas = %s" % alphas)

        uppath = './ALPHA'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(alphas)):
            np.random.set_state(state)
            alpha = alphas[i]
            print("Generating clouds for alpha %.3g" % alpha)
            path = './%.3g' % alpha
            cloud_fil = 'clouds_alpha_%.3g.h5' % alpha
            #print(path)
            #print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['alpha'] = alpha
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')


    elif (flg==9):

        ## zeta loop
        zetas = np.array([-5.0, -2.0, -1.0, 0.0, 2.0])

        print("Looping over zetas = %s" % zetas)

        uppath = './ZETA'
        if not os.path.exists(uppath):
            os.makedirs(uppath)
        os.chdir(uppath)
        
        for i in range(len(zetas)):
            np.random.set_state(state)
            zeta = zetas[i]
            print("Generating clouds for zeta %.3g" % zeta)
            path = './%.3g' % zeta
            cloud_fil = 'clouds_zeta_%.3g.h5' % zeta
            print(path)
            print(cloud_fil)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['zeta'] = zeta
            create_and_plot_clouds(params, cloud_fil=cloud_fil)
        
            os.chdir('..')

        os.chdir('..')

        

def makeall_rays(updir='MCLMIN'):

    os.chdir(os.path.join('.', updir))
    dirs = os.listdir(path='.')

    for dir in dirs:
        os.chdir(dir)
        clouds_fil = glob.glob('clouds*.h5')[0]
        namestr = clouds_fil[6:-3]
        rays_fil = 'rays%s.h5' % namestr

        create_rays(clouds_fil,rays_fil)
        os.chdir('..')

    os.chdir('..')


def create_rays(clouds_fil, rays_fil):
    ## Taken from make_rays.py

    clouds = Clouds()
    clouds.load(clouds_fil)
    params = clouds.params
    np.random.seed(seed=params['seed'])

    ## Generate and Plot Spectra for N random rays passing through domain
    ## Also plot distribution of clouds and rays

    # Create N random Rays passing through domain
    N = 10000
    sg = SpectrumGenerator(clouds, subcloud_turb=True)
    rays = sg.generate_ray_sample(N, params['center'],  params['dclmax'],
                                  first_n_to_plot=50, attach_spectra=True)
    rays.save(rays_fil)

    # Create histograms of EW, clouds intersected, and column density of all rays
    plot_histogram(rays.column_densities, "column densities", clouds, log=True, xrange=[11,18])
    plot_histogram(rays.EWs, "equivalent widths", clouds, xrange=[0, 1.0], n_bins=40)
            
        


def run_quad_plot(updir='MCLMIN'):

   
    os.chdir(os.path.join('.', updir))
    allfils = os.listdir(path='.')
    dirs = []
    for fils in allfils:
        if os.path.isdir(fils):
            dirs.append(fils)


    clouds_list = []
    rays_list = []

    mclmin_list = []
    ncl_list = []
    mtot_list = []
    dclmax_list = []
    Zcl_list = []
    vmax_list = []
    beta_list = []
    alpha_list = []
    zeta_list = []

    flgs = {}
    flgs['mclmin'] = False
    flgs['ncl'] = False
    flgs['mtot'] = False
    flgs['dclmax'] = False
    flgs['Zcl'] = False
    flgs['vmax'] = False
    flgs['beta'] = False
    flgs['alpha'] = False
    flgs['zeta'] = False
    
    for dir in dirs:
        os.chdir(dir)
        clouds_fil = glob.glob('clouds*.h5')[0]
        namestr = clouds_fil[6:-3]
        rays_fil = 'rays%s.h5' % namestr

        rays = Rays(generate=False)
        rays.load(rays_fil)
        rays_list.append(rays)
        clouds = Clouds()
        clouds.load(clouds_fil)
        clouds_list.append(clouds)

        p = clouds.params
        mclmin_list.append(p['mclmin'])
        ncl_list.append(p['n_cl'])
        mtot_list.append(p['total_mass'])
        dclmax_list.append(p['dclmax'])
        Zcl_list.append(p['Z_cl'])
        vmax_list.append(p['vmax'])
        beta_list.append(p['beta'])
        alpha_list.append(p['alpha'])
        zeta_list.append(p['zeta'])

        os.chdir('..')


    if(updir=='MCLMIN'):

        flgs['mclmin'] = True

        sort = np.argsort(mclmin_list)
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_mclmin.png'


    elif(updir=='NCL'):

        flgs['ncl'] = True
       
        sort = np.argsort(ncl_list)
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_ncl.png'
       

    elif(updir=='MTOT'):

        flgs['mtot'] = True
        sort = np.flip(np.argsort(mtot_list))
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_mtot.png'

    elif(updir=='DCLMAX'):

        flgs['dclmax'] = True
        sort = np.argsort(dclmax_list)
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_dclmax.png'

    elif(updir=='ZCL'):

        flgs['Zcl'] = True
        sort = np.flip(np.argsort(Zcl_list))
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_Zcl.png'

    elif(updir=='VMAX'):

        flgs['vmax'] = True
        sort = np.flip(np.argsort(vmax_list))
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_vmax.png'

    elif(updir=='BETA'):

        flgs['beta'] = True
        sort = np.argsort(beta_list)
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_beta.png'

    elif(updir=='ALPHA'):

        flgs['alpha'] = True
        sort = np.flip(np.argsort(alpha_list))
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_alpha.png'

    elif(updir=='ZETA'):

        flgs['zeta'] = True
        sort = np.argsort(zeta_list)
        sort = sort.tolist()
        s_clouds_list = []
        s_rays_list = []
        for i in sort:
            s_clouds_list.append(clouds_list[i])
            s_rays_list.append(rays_list[i])
        ofil = 'quad_zeta.png'
        

    make_quad_plot(s_rays_list, s_clouds_list, filename=ofil, flgs_dict=flgs)
    os.chdir('..')


if __name__ == '__main__':

    #makeall_clouds(flg=1)
    makeall_rays()
    #run_quad_plot(updir='ZETA')
