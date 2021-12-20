
import bempp.api, numpy as np, time, os, matplotlib.pyplot as plt
from math import pi
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from matplotlib import pylab as plt
from numba import jit
import trimesh

# This python must be saved in a directory where you have a folder named
# /Molecule/Molecule_Name, as obviusly Molecule_Name holds a .pdb or .pqr file
# ; otherwise, this function won't do anything.

# Data is saved in format {mol_name}_{mesh_density}-{it_count}
# Where mol_name is the abreviated name of the molecule
# mesh_density is the density of the mesh in elements per square amstrong
# it_count is for the mesh ref pluggin and will be treaten as -0 when is the first grid made

# IMPORTANT BUGS - 1. NANOSHAPER MUST BE REPAIRED - ONLY MSMS ALLOWED
# 2. print('juan') not printing in xyzr_to_msh !!!!!!!!!!!! This may be a reason
#    why .msh file is not being created.

#With the .pdb file we can build .pqr & .xyzr files, and they don't change when the mesh density is changed.

def face_and_vert_to_off(face_array , vert_array , path , file_name ):
    '''
    Creates off file from face and vert arrays.
    '''
    off_file = open( os.path.join( path , file_name + '.off' ) , 'w+')
    
    off_file.write( 'OFF\n' )
    off_file.write( '{0} {1} 0 \n'.format(len(vert_array) , len(face_array)) )
    for vert in vert_array:
        off_file.write( str(vert)[1:-1] +'\n' )
    
    for face in face_array:
        off_file.write( '3 ' + str(face - 1)[1:-1] +'\n' )
        
    off_file.close()
    
    return None

def Improve_Mesh(face_array , vert_array , path , file_name ):
    '''
    Executes ImproveSurfMesh and substitutes files.
    '''
    
    #os.system('export LD_LIBRARY_PATH=/vicenteramm/lib/')
    face_and_vert_to_off(face_array , vert_array , path , file_name)
    
    #Improve_surf_Path = '/home/vicenteramm/Software/fetk/gamer/tools/ImproveSurfMesh/ImproveSurfMesh'
    Improve_surf_Path = '/home/chris/Software/fetk/gamer/tools/ImproveSurfMesh/ImproveSurfMesh'
    os.system( Improve_surf_Path + ' --smooth --correct-normals ' + os.path.join(path , file_name +'.off')  )
    
    os.system('mv  {0}/{1}'.format(path, file_name + '_improved_0.off ') + 
                                 '{0}/{1}'.format(path, file_name + '.off '))
    
    new_off_file =  open( os.path.join( path , file_name + '.off' ) , 'r').read().split('\n')
    #print(new_off_file)
    
    num_verts = int(new_off_file[1].split()[0])
    num_faces = int(new_off_file[1].split()[1])

    new_vert_array = np.empty((0,3))
    for line in new_off_file[2:num_verts+2]:
        new_vert_array = np.vstack((new_vert_array , line.split() ))


    new_face_array = np.empty((0,3))
    for line in new_off_file[num_verts+2:-1]:
        new_face_array = np.vstack((new_face_array , line.split()[1:] ))


    new_vert_array = new_vert_array.astype(float)
    new_face_array = new_face_array.astype(int  ) + 1
    
    return new_face_array , new_vert_array

def pdb_to_pqr(mol_name , stern_thickness , method = 'amber' ):
    '''
    Function that makes .pqr file from .pdb using Software/apbs-pdb2pqr-master/pdb2pqr/main.py
    Be careful of the version and the save directory of the pdb2pqr python shell.
    mol_name : Abreviated name of the molecule
    stern_thicness : Length of the stern layer
    method         : This parameter is an 
    '''
    path = os.getcwd()
        
    pdb_file , pdb_directory = mol_name+'.pdb' , os.path.join('Molecule',mol_name)
    pqr_file , xyzr_file     = mol_name+'.pqr' , mol_name+'.xyzr'
    
    if os.path.isfile(os.path.join('Molecule',mol_name,pqr_file)):
        print('File already exists in directory.')
        return None
    
    # The apbs-pdb2pqr rutine, allow us to generate a .pqr file
    pdb2pqr_dir = os.path.join('Software','apbs-pdb2pqr-master','pdb2pqr','main.py')
    exe=('python2.7  ' + pdb2pqr_dir + ' '+ os.path.join(pdb_directory,pdb_file) +
         ' --ff='+method+' ' + os.path.join(pdb_directory,pqr_file)   )
    
    os.system(exe)
    
    # Now, .pqr file contains unneeded text inputs, we will save the rows starting with 'ATOM'.
    
    pqr_Text = open( os.path.join(pdb_directory , pqr_file) ).read()
    pqr_Text_xyzr = open(os.path.join(pdb_directory , xyzr_file )  ,'w+')

    
    for i in pqr_Text.split('\n'):
        row=i.split()
        if row[0]=='ATOM':
            aux=row[5]+' '+row[6]+' '+row[7]+' '+row[-1]
            pqr_Text_xyzr.write(aux + '\n')   
    pqr_Text_xyzr.close()
    
    print('Global .pqr & .xyzr ready.')
    
    # The exterior interface is easy to add, by increasing each atom radii
    if stern_thickness>0: 
        xyzr_file_stern = os.path.join(pdb_directory , mol_name +'_stern.xyzr')
        pqr_Text_xyzr_s = open(xyzr_file_stern ,'w')
        
        for i in pqr_Text.split('\n'):
            row=i.split()
            if row[0]=='ATOM':
                R_vv=float(row[-1])+stern_thickness
                pqr_Text_xyzr_s.write(row[5]+' '+row[6]+' '+row[7]+' '+str(R_vv)+'\n' )      
        pqr_Text_xyzr_s.close()
        print('Global _stern.pqr & _stern.xyzr ready.')
    
    return 

def pqr_to_xyzr(mol_name , stern_thickness , method = 'amber' ):
    '''
    Extracts .xyzr information from .pqr
    mol_name : Abreviated name of the molecule
    stern_thickness : Length of the stern layer
    method          : amber by default , a pdb2pqr parameter to build the mesh.
    '''
    path = os.getcwd()
    
    pqr_directory = os.path.join('Molecule',mol_name)
    pqr_file , xyzr_file     = mol_name+'.pqr' , mol_name+'.xyzr'
     
    # Now, .pqr file contains unneeded text inputs, we will save the rows starting with 'ATOM'.
    
    pqr_Text = open( os.path.join(pqr_directory , pqr_file) ).read()
    pqr_Text_xyzr = open(os.path.join(pqr_directory , xyzr_file )  ,'w+')

    
    for i in pqr_Text.split('\n'):
        row=i.split()
        if len(row)==0: continue
            
        if row[0]=='ATOM':
            aux=' '.join( [row[5],row[6],row[7],row[-1]] )
            pqr_Text_xyzr.write(aux + '\n')   
    pqr_Text_xyzr.close()
    
    print('.xyzr File from .pqr ready.')
    
    # The exterior interface is easy to add, by increasing each atom radii
    if stern_thickness>0: 
        xyzr_file_stern = os.path.join(pqr_directory , mol_name +'_stern.xyzr')
        pqr_Text_xyzr_s = open(xyzr_file_stern ,'w')
        
        for i in pqr_Text.split('\n'):
            row=i.split()
            if row[0]=='ATOM':
                R_vv=float(row[-1])+stern_thickness
                pqr_Text_xyzr_s.write(row[5]+' '+row[6]+' '+row[7]+' '+str(R_vv)+'\n' )      
        pqr_Text_xyzr_s.close()
        print('Global _stern.pqr & _stern.xyzr ready.')
        
    return None

def NanoShaper_config(xyzr_file , dens , probe_radius):
    '''
    Yet in beta version. Changes some data to build the mesh with NanoShaper
    xyzr_file : Directory of the xyzr_file
    dens      : mesh density
    probe_radius : might be set to 1.4
    '''
    t1 = (  'Grid_scale = {:s}'.format(str(dens)) 
                #Specify in Angstrom the inverse of the side of the grid cubes  
              , 'Grid_perfil = 80.0 '                     
                #Percentage that the surface maximum dimension occupies with
                # respect to the total grid size,
              , 'XYZR_FileName = {:s}'.format(xyzr_file)  
              ,  'Build_epsilon_maps = false'              
              , 'Build_status_map = false'                
              ,  'Save_Mesh_MSMS_Format = true'            
              ,  'Compute_Vertex_Normals = true'           
              ,  'Surface = ses  '                         
              ,  'Smooth_Mesh = true'                      
              ,  'Skin_Surface_Parameter = 0.45'           
              ,  'Cavity_Detection_Filling = false'        
              ,  'Conditional_Volume_Filling_Value = 11.4' 
              ,  'Keep_Water_Shaped_Cavities = false'      
              ,  'Probe_Radius = {:s}'.format( str(probe_radius) )                
              ,  'Accurate_Triangulation = true'           
              ,  'Triangulation = true'                    
              ,  'Check_duplicated_vertices = true'        
              ,  'Save_Status_map = false'                 
              ,  'Save_PovRay = false'                     )
    return t1

def xyzr_to_msh(mol_name , dens , probe_radius , stern_thickness , min_area , Mallador ,
               suffix = '' , build_msh=True):
    '''
    Makes msh (mesh format for BEMPP) from xyzr file
    mol_name : Abreviated name of the molecule
    dens     : Mesh density
    probe_radius : might be set to 1.4[A]
    stern_thickness : Length of the stern layer
    min_area        : Discards elements with less area than this value
    Mallador        : MSMS or NanoShaper
    
    outputs : Molecule/{mol_name}/{mol_name}_{dens}-0.msh
    Where -0 was added because of probable future mesh refinement and easier handling of meshes.
    '''

    path = os.getcwd()
    mol_directory = os.path.join('Molecule',mol_name)
    path = os.path.join(path , mol_directory)
    xyzr_file     = os.path.join(mol_directory, mol_name + '.xyzr') 
    
    if stern_thickness > 0:  xyzr_s_file = os.path.join(mol_directory , mol_name + '_stern.xyzr'  )
    
    # The executable line must be:
    #  path/Software/msms/msms.x86_64Linux2.2.6.1 
    # -if path/mol_name.xyzr       (Input File)
    # -of path/mol_name -prob 1.4 -d 3.    (Output File)
   
    # The directory of msms/NS needs to be checked, it must be saved in the same folder that is this file
    if Mallador == 'MSMS':  
        #Usar M_Path = 'msms' en caso de tener msms instalado, si no lo reconoce usará malla importada previamente.
        M_path = os.path.join('Software','msms','msms.x86_64Linux2.2.6.1')                                        
        #M_path = 'msms' 
        mode= ' -no_header'
        prob_rad, dens_msh = ' -prob ' + str(probe_radius), ' -d ' + str(dens)
        exe= (M_path
              +' -if ' + xyzr_file
              +' -of ' + os.path.join(mol_directory , mol_name )+'_{0:s}-0'.format( str(dens) )
              + prob_rad  + dens_msh + mode )
        os.system(exe)
        print('Normal .vert & .face Done')

        grid = factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador , suffix = '-0', build_msh=build_msh)
        print('Normal .msh Done')
        
        # As the possibility of using a stern layer is available:
        if stern_thickness > 0:
            prob_rad, dens_msh = ' -prob ' + str(probe_radius), ' -d ' + str(dens)
            exe= (M_path+' -if '  + xyzr_s_file + 
              ' -of ' + mol_directory + mol_name +'_stern' + prob_rad  + dens_msh  + mode )
            os.system(exe)
            print('Stern .vert & .face Done')
            stern_grid= factory_fun_msh( mol_directory , mol_name+'_stern', min_area )
            print('Stern .msh Done')
        
    elif Mallador == 'NanoShaper': 
        Ubication = os.path.join('Software','NanoShaper','NanoShaper')
        config  = os.path.join('Software','NanoShaper','config')
        
        # NanoShaper options can be changed from the config file
        Config_text = open(config,'w')
        
        Conf_text = NanoShaper_config(xyzr_file , dens , probe_radius)
        
        Config_text.write('\n'.join(Conf_text))
        Config_text.close()
        
        # Files are moved to the same directory
        os.system(' '.join( ('./'+Ubication,config)  ))
        
        Files = ('triangleAreas{0:s}.txt','triangulatedSurf{0:s}.face' ,'triangulatedSurf{0:s}.vert',
         'exposedIndices{0:s}.txt','exposed{0:s}.xyz  ' , 'stderror{0:s}.txt' )
        for f in Files:

            if f[-4:]=='vert':
                os.system(' '.join( ('mv ', f.format('') 
                                 , os.path.join(mol_directory,'{0}_{1}{2}.vert'.format(mol_name , str(dens), suffix)))))
            if f[-4:]=='face':
                os.system(' '.join( ('mv ', f.format('')  
                                 , os.path.join(mol_directory,'{0}_{1}{2}.face'.format(mol_name , str(dens), suffix)))))
            
        if not os.path.isfile(os.path.join(path , '{0:s}{1:s}{2:s}.vert'.format(mol_name, '_'+str(dens) , suffix))):
            print('Fatal error : Vert file not created.')
            
        
        print('Normal .vert & .face Done')
        
        #checked until here, everything looks alright
        grid = factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador , suffix = '-0',build_msh=build_msh)
        print('Loaded the grid (.msh format)')
        
        if stern_thickness>0:
            # NanoShaper options can be changed only in the config file
            Config_text = open(config,'w')
            Conf_text = NanoShaper_config(xyzr_file_stern , dens , probe_radius)
            
            Config_text.write('\n'.join(Conf_text))
            Config_text.close()

            # Files are moved to the same directory
            os.system(' '.join( ('./'+Mallador,config)  ))

            Files = ('triangleAreas{0:s}.txt','triangulatedSurf{0:s}.face' ,'triangulatedSurf{0:s}.vert',
             'exposedIndices{0:s}.txt','exposed{0:s}.xyz  ' , 'stderror{0:s}.txt' )
            for f in Files:
                os.system(' '.join( ('mv ', f.format('') 
                                     , os.path.join(mol_directory,f.format('_stern_'+str(dens)))))) 
            print('Stern .vert & .face Done')
            
            stern_grid= factory_fun_msh( mol_directory , mol_name+'_stern', min_area , dens , Mallador = Mallador)
            print('stern_.msh File Done')
    print('Mesh Ready')
    return

def factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador , suffix = '', build_msh = True):
    '''
    This functions builds msh file adding faces and respective vertices.
    mol_directory : Directory of the molecule
    mol_name      : Abreviated name of the molecule
    min_area      : Min. area set to exclude small elements
    dens          : mesh density
    Mallador      : MSMS - NanoShaper or Self (if doing the GOMR)
    suffix        : Suffix of the .vert and .face file after the mesh density ({mol_name}_{d}{suffix})
                    might be used as -{it_count}
    '''
    # .vert and .face files are readed    
    if Mallador == 'MSMS':
        print('Loading the MSMS grid.')
        vert_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix) ) , usecols=(0,1,2))
        face_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.face'.format(str(dens),suffix) ) , dtype=int , usecols=(0,1,2)) -1
        grid = bempp.api.Grid (np.transpose(vert_Text), np.transpose(face_Text))
        
    elif Mallador == 'NanoShaper':
        print('Loading the NanoShaper grid.')
        vert_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix) ) , skiprows=3 , usecols=(0,1,2) )
        face_Text = np.loadtxt(os.path.join(mol_directory , mol_name +'_{0:s}{1}.face'.format(str(dens),suffix) ) , skiprows=3 , usecols=(0,1,2) , dtype=int)-1
        grid = bempp.api.Grid (np.transpose(vert_Text), np.transpose(face_Text))
        
    elif Mallador == 'Self':
        print('Loading the built grid.')
        vert_Text = np.loadtxt( os.path.join(mol_directory , mol_name +'_{0:s}{1}.vert'.format(str(dens),suffix)), usecols=(0,1,2) )
        face_Text = np.loadtxt( os.path.join(mol_directory , mol_name +'_{0:s}{1}.face'.format(str(dens),suffix)), usecols=(0,1,2) , dtype=int)-1
        grid = bempp.api.Grid (np.transpose(vert_Text), np.transpose(face_Text))
    
    export_file = os.path.join(mol_directory , mol_name +'_'+str(dens)+ suffix +'.msh' )
    print(export_file)
    bempp.api.export(export_file, grid=grid) ##cambiar
    
    return grid

def triangle_areas(mol_directory , mol_name , dens , return_data = False , suffix = '', Self_build = False):
    """
    This function calculates the area of each element.
    Avoid using this with NanoShaper, only MSMS recomended
    Self_build : False if using MSMS or NanoShaper - True if building with new methods
    Has a BUG! probably not opening .vert or .face or not creating .txt or both :P .
    """
    
    vert_Text = open( os.path.join(mol_directory , mol_name +'_'+str(dens)+suffix+'.vert' ) ).read().split('\n')
    face_Text = open( os.path.join(mol_directory , mol_name +'_'+str(dens)+suffix+'.face' ) ).read().split('\n')
    area_list = np.empty((0,1))
    area_Text = open( os.path.join(mol_directory , 'triangleAreas_'+str(dens)+suffix+'.txt' ) , 'w+')
    
    vertex = np.empty((0,3))
    
    if not Self_build:
        for line in vert_Text:
            line = line.split()
            if len(line) !=9: continue
            vertex = np.vstack(( vertex, np.array(line[0:3]).astype(float) ))

        atotal=0.0
        # Grid assamble
        for line in face_Text:
            line = line.split()
            if len(line)!=5: continue
            A, B, C, _, _ = np.array(line).astype(int)
            side1, side2  = vertex[B-1]-vertex[A-1], vertex[C-1]-vertex[A-1]
            face_area = 0.5*np.linalg.norm(np.cross(side1, side2))

            area_Text.write( str(face_area)+'\n' )

            area_list = np.vstack( (area_list , face_area ) )
            atotal += face_area

        area_Text.close()

        if return_data:
            return area_list
        
    elif Self_build:
        
        for line in vert_Text[:-1]:
            line = line.split()
            
            vertex = np.vstack(( vertex, np.array(line[0:3]).astype(float) ))

        atotal=0.0
        # Grid assamble
        for line in face_Text[:-1]:
            line = line.split()
            A, B, C = np.array(line[0:3]).astype(int)
            side1, side2  = vertex[B-1]-vertex[A-1], vertex[C-1]-vertex[A-1]
            face_area = 0.5*np.linalg.norm(np.cross(side1, side2))
            area_Text.write( str(face_area)+'\n' )

            area_list = np.vstack( (area_list , face_area ) )
            atotal += face_area

        area_Text.close()

        if return_data:
            return area_list
    
    return None

def normals_to_element( face_array , vert_array , check_dir = False ):
    '''
    Calculates normals to a given element, pointint outwards.
    face_array : Array of vertex position for each triangle
    vert_array : Array of vertices
    check_dir  : checks direction of normals. WORKS ONLY FOR A SPHERE WITH RADII 1!!!!!!!!!!!!
    '''

    normals = np.empty((0,3))
    element_cent = np.empty((0,3))
    
    check_list = np.empty((0,1))
    
    for face in face_array:
        
        f1,f2,f3 = face-1
        v1 , v2 , v3 = vert_array[f1] , vert_array[f2] , vert_array[f3]
        n = np.cross( v2-v1 , v3-v1 ) 
        normals = np.vstack((normals , n/np.linalg.norm(n) )) 
        element_cent = np.vstack((element_cent, (v1+v2+v3)/3. ))
        
        if check_dir:
            v_c = v1 + v2 + v3
            pdot= np.dot( v_c , n )
            if pdot>0:
                check = True
            else:
                check = False
            check_list = np.vstack( (check_list , check ) )
            

    return normals , check_list[:,0]


def vert_and_face_arrays_to_text_and_mesh(mol_name , vert_array , face_array , suffix 
                                          , dens=2.0 , Self_build=True):
    '''
    This rutine saves the info from vert_array and face_array and creates .msh and areas.txt files
    mol_name : Abreviated name for the molecule
    dens     : Mesh density, anyway is not a parameter, just a name for the file
    vert_array: array containing verts
    face_array: array containing verts positions for each face
    suffix    : text added to diference the meshes.
    
    Returns None but creates Molecule/{mol_name}/{mol_name}_{mesh_density}{suffix}.msh file.
    '''
    normalized_path = os.path.join('Molecule',mol_name,mol_name+'_'+str(dens)+suffix)
    
    vert_txt = open( normalized_path+'.vert' , 'w+' )
    for vert in vert_array:
        txt = ' '.join( vert.astype(str) )
        vert_txt.write( txt + '\n')
    vert_txt.close()
    
    face_txt = open( normalized_path+'.face' , 'w+' )
    for face in face_array:
        txt = ' '.join( face.astype(int).astype(str) )
        face_txt.write( txt + '\n')
    face_txt.close()
    
    mol_directory = os.path.join('Molecule',mol_name)
    min_area = 0

    factory_fun_msh( mol_directory , mol_name , min_area , dens , Mallador='Self', suffix=suffix)
    #triangle_areas(mol_directory , mol_name , str(dens) , suffix = suffix , Self_build = Self_build)
    
    return None
@jit
def Grid_loader(mol_name , mesh_density , suffix , Mallador , GAMer=False, build_msh = True):
    
    path = os.path.join('Molecule',mol_name)
    grid_name_File =  os.path.join(path,mol_name + '_'+str(mesh_density)+suffix+'.msh')
    
    print(grid_name_File)
    
    if os.path.isfile(grid_name_File) and suffix == '-0':
        
        pqr_directory = os.path.join('Molecule',mol_name, mol_name+'.pqr' )
        
        if not os.path.isfile(pqr_directory):
            pdb_to_pqr(mol_name , stern_thickness , method = 'amber' )
       
    if suffix == '-0':
        pqr_to_xyzr(mol_name , stern_thickness=0 , method = 'amber' )
        xyzr_to_msh(mol_name , mesh_density , 1.5 , 0 , 0
                    , Mallador, suffix = suffix , build_msh = build_msh)
    if not build_msh:
        
        return None    
    
    #print('Working on '+grid_name_File )
    grid = bempp.api.import_grid(grid_name_File)
    
    
    
    if GAMer:
        face_array = np.transpose(grid.elements)+1
        vert_array = np.transpose(grid.vertices)
        
        new_face_array , new_vert_array = Improve_Mesh(face_array , vert_array , path , mol_name + '_'+str(mesh_density)+suffix )
        
        vert_and_face_arrays_to_text_and_mesh(mol_name , new_vert_array , new_face_array , suffix 
                                          , dens=mesh_density , Self_build=True)
        
        grid = bempp.api.import_grid(grid_name_File)
    
    return grid

from constants import *
def fine_grid_maker(mol_name , dens_f=40.0):
    '''
    Does a 40.0 grid
    Input 
    mol_name : Name of the molecule
    dens     : Mesh density
    Output
    None
    '''
    path = os.path.join('Molecule' , mol_name , mol_name + '_{0:.1f}'.format(dens_f))
    if os.path.isfile( path + '.vert' ):
        return None        
    

    x_q , q = run_pqr(mol_name)
    Grid_loader( mol_name , dens_f , '-0' , 'MSMS' , GAMer = False  , build_msh = False)
    
    return None

def run_pqr(mol_name):
    
    global q , x_q
    
    path = os.path.join('Molecule',mol_name) 
    
    q, x_q = np.empty(0), np.empty((0,3))
    pqr_file = os.path.join(path,mol_name+'.pqr')
    charges_file = open( pqr_file , 'r').read().split('\n')

    for line in charges_file:
        line = line.split()
        if len(line)==0: continue
        if line[0]!='ATOM': continue
        q = np.append( q, float(line[8]))
        x_q = np.vstack( ( x_q, np.array(line[5:8]).astype(float) ) )  

    return q , x_q

def error_test(dif, grid, q, x_q):
    
    '''
    Calculate maximum error, maximum potential, maximum ratio, maximum area in maximum error triangle
    Generates a matrix with all potentials, another with all ratio's.
    This is to select the criterion that most influences has in the error.
    
    Parameters:
    dif: error matrix
    grid
    q
    x_q
    
    returns:
    error, area, ratio and potential from maximum error triangle
    '''
    global ep_m
    total_error = np.abs(np.sum (dif))
    print ('Total Error is: {0:.7f}'.format(total_error))
    error_max = np.max(dif)
    print ('Maximum Error is: {0:.7f}'.format(error_max))
    index_error_max = np.where(np.abs(error_max-dif)<1e-12)[0]
    vert_max = np.transpose(grid.vertices)[np.transpose(grid.elements)[index_error_max]]
    #print (np.transpose(grid.elements)[index_error_max])
    #print (index_error_max)
    #print (vert_max)
    error_max_area = grid.volumes[index_error_max]
    print ('Area in max error triangle is:', error_max_area)
    #print (np.sum(grid.volumes))
    all_area = grid.volumes
            
    triangles = (np.transpose(grid.vertices)[np.transpose(grid.elements)]) 
            
    #ratio
    ratio = np.empty(grid.number_of_elements)
    for i in range (grid.number_of_elements):
        L_ab = triangles[i][1] - triangles[i][0]
        L_bc = triangles[i][2] - triangles[i][1]
        L_ca = triangles[i][0] - triangles[i][2]
        A = np.linalg.norm(np.cross(L_ab, L_bc))/np.linalg.norm(L_bc)
        B = np.linalg.norm(np.cross(L_bc, L_ca))/np.linalg.norm(L_ca)
        C = np.linalg.norm(np.cross(L_ca, L_ab))/np.linalg.norm(L_ab)
        values = np.array([A,B,C])
        h_max = np.max(values)
        h_min = np.min(values)
        ratio[i] = h_max / h_min #all ratio
                
    L_ab = vert_max[0][1] - vert_max[0][0]
    L_bc = vert_max[0][2] - vert_max[0][1]
    L_ca = vert_max[0][0] - vert_max[0][2]
            
    A = np.linalg.norm(np.cross(L_ab, L_bc))/np.linalg.norm(L_bc)
    B = np.linalg.norm(np.cross(L_bc, L_ca))/np.linalg.norm(L_ca)
    C = np.linalg.norm(np.cross(L_ca, L_ab))/np.linalg.norm(L_ab)
    values = np.array([A,B,C])
    h_max = np.max(values)
    h_min = np.min(values)
            
    print ('Distance Ratio in max_error_triangle:', h_max/h_min)
            
    #potential
    pot = np.empty(grid.number_of_elements)
    variable = np.empty(len(q))
    for i in range (grid.number_of_elements):
        x = (triangles[i][0][0] + triangles[i][1][0] + triangles[i][2][0]) / 3
        y = (triangles[i][0][1] + triangles[i][1][1] + triangles[i][2][1]) / 3
        z = (triangles[i][0][2] + triangles[i][1][2] + triangles[i][2][2]) / 3
        r_c = np.array([x,y,z])
        for j in range (len(q)):
            variable[j] = q[j] / (4*np.pi*ep_m*np.linalg.norm(r_c - x_q[j]))
        pot[i] = np.sum(variable) #all potentials
        
    x = (vert_max[0][0][0] + vert_max[0][1][0] + vert_max[0][2][0]) / 3
    y = (vert_max[0][0][1] + vert_max[0][1][1] + vert_max[0][2][1]) / 3
    z = (vert_max[0][0][2] + vert_max[0][1][2] + vert_max[0][2][2]) / 3
    r_center = np.array([x,y,z])
    pot_max = np.sum (q / (4*np.pi*ep_m*np.linalg.norm(r_center - x_q, axis = 1)))
    print ('Potential in max_error_triangle:',(pot_max))
            
    error_sort = np.argsort(np.abs(dif))

    potential_sort = np.argsort(np.abs(pot))
    #np.savetxt ('error', error_sort[::-1], fmt='%10.0f')
    #np.savetxt ('potential', potential_sort[::-1], fmt='%10.0f')
    ratio_sort = np.argsort(np.abs(ratio))
    area_sort = np.argsort(np.abs(all_area))
    pot_ratio_sort = np.argsort (np.abs(pot*ratio))
    pot_area_sort = np.argsort (np.abs(pot*all_area))
    ratio_area_sort = np.argsort (np.abs(all_area*ratio))
            
    div_pot_ratio = np.argsort (np.abs(pot/ratio))
    div_ratio_pot = np.argsort (np.abs(ratio/pot))
    div_area_ratio = np.argsort (np.abs(all_area/ratio))
    div_ratio_area = np.argsort (np.abs(ratio/all_area))
    div_pot_area = np.argsort (np.abs(pot/all_area))
    div_area_pot = np.argsort (np.abs(all_area/pot))
    ####
    error_pot = np.linalg.norm(error_sort - potential_sort)
    error_ratio = np.linalg.norm(error_sort - ratio_sort)
    error_area = np.linalg.norm(error_sort - area_sort)
    error_pot_ratio = np.linalg.norm(error_sort - pot_ratio_sort)
    error_pot_area = np.linalg.norm(error_sort - pot_area_sort)
    error_ratio_area = np.linalg.norm(error_sort - ratio_area_sort)
            
    error_pot_ratio2 = np.linalg.norm(error_sort - div_pot_ratio)
    error_ratio_pot = np.linalg.norm(error_sort - div_ratio_pot)
    error_area_ratio = np.linalg.norm(error_sort - div_area_ratio)
    error_ratio_area2 = np.linalg.norm(error_sort - div_ratio_area)
    error_pot_area2 = np.linalg.norm(error_sort - div_pot_area)
    error_area_pot = np.linalg.norm(error_sort - div_area_pot)
            
            
    #print ('Euclidean distance for error vs potential is:', error_pot)
    #print ('Euclidean distance for error vs ratio is:',error_ratio)
    #print ('Euclidean distance for error vs area is:',error_area)
    #print ('Euclidean distance for error vs pot*ratio is:',error_pot_ratio)
    #print ('Euclidean distance for error vs pot*area is:',error_pot_area)
    #print ('Euclidean distance for error vs ratio*area is:',error_ratio_area)
            
    #print ('Euclidean distance for error vs pot/ratio is:',error_pot_ratio2)
    #print ('Euclidean distance for error vs ratio/pot is:',error_ratio_pot)
    #print ('Euclidean distance for error vs area/ratio is:',error_area_ratio)
    #print ('Euclidean distance for error vs ratio/area is:',error_ratio_area2)
    #print ('Euclidean distance for error vs pot/area is:',error_pot_area2)
    #print ('Euclidean distance for error vs area/pot is:',error_area_pot)
    
    Array = np.array([error_pot, error_ratio, error_area, error_pot_ratio, error_pot_area, error_ratio_area,
                     error_pot_ratio2, error_ratio_pot, error_area_ratio, error_ratio_area2, error_pot_area2,
                     error_area_pot])
    #print ('Potential, Ratio, Area, Pot*Ratio, Pot*area, Ratio*Area,\
 #Pot/Ratio, Ratio/Pot, Area/Ratio, Ratio/Area, Pot/Area, Area/Pot')
    
    #print (np.argsort(Array)[0], np.argsort(Array)[1], np.argsort(Array)[2], np.argsort(Array)[3])
    
    
    return error_max, error_max_area, h_max/h_min, pot_max, pot

def potential_calc(grid, q , x_q ):
    '''
    This function calculates the potential of all elements
    
    Parameters:
    grid
    q
    x_q
    
    '''
    triangles = (np.transpose(grid.vertices)[np.transpose(grid.elements)]) 
    pot = np.empty(grid.number_of_elements)
    variable = np.empty(len(q))
    for i in range (grid.number_of_elements):
        x = (triangles[i][0][0] + triangles[i][1][0] + triangles[i][2][0]) / 3
        y = (triangles[i][0][1] + triangles[i][1][1] + triangles[i][2][1]) / 3
        z = (triangles[i][0][2] + triangles[i][1][2] + triangles[i][2][2]) / 3
        r_c = np.array([x,y,z])
        for j in range (len(q)):
            variable[j] = q[j] / (4*np.pi*ep_m*np.linalg.norm(r_c - x_q[j]))
        pot[i] = np.sum(variable) #all potentials
    
    return pot
