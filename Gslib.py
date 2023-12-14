# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:38:46 2019
    
    Gslib class and methods
    
@author: Roberto
"""
import pandas
import numpy
import pdb
class Gslib:
    def __intit__(self):
        self.filename       ='Null'
        self.title          ='Null'
        self.n_properties   =0
        self.prop_names     =[]
        self.n_lines_head   =0
        self.nx             =0
        self.ny             =0
        self.nz             =0
        self.data           =[]


    def __repr__(self):
        if self.n_properties>1:
            return f"Gslib file\nFile name: {self.filename}\n{self.n_properties} properties: {self.prop_names}"
        elif self.n_properties==1:
            return f"Gslib file\nFile name: {self.filename}\nOne property: {self.prop_names}"
        else:
            return "Gslib object with no data."


    def Gslib_read(self, filename, order_z='no', overwrite_file= 'no', zmax=0):
        #Reads a Gslib file and creates a Gslib Instance
        
        self.filename= filename
        
        f= open(self.filename, 'r+')
        self.title= f.readline()
    
        self.n_properties= int(f.readline().split()[0])
    
        self.prop_names=[]
        for i in range(self.n_properties):
            self.prop_names.append(f.readline().split()[0])
    
        self.n_lines_head = self.n_properties + 2
        
        self.data = pandas.read_csv(filepath_or_buffer=self.filename, header=None, 
                                    names = self.prop_names, skiprows=self.n_lines_head, 
                                    delim_whitespace=True)
        
        if order_z=='yes':
            
            self.data[self.prop_names[2]]= self.data[self.prop_names[2]].mul(-1).add(zmax).add(1)
            
        if overwrite_file == 'yes':
            f.seek(0)
            f.write(self.title)
            f.write(f'{self.n_properties}\n')
        
            for i in self.prop_names:
                f.write(f'{i}\n')
            
            f.truncate()
            self.data.to_csv(self.filename,sep="\t",index=False,header=False,mode='a')
        
        f.close()
        return self



    def Gslib_write(self, filename, prop_names, data, nx, ny, nz, folder, title='null'):
        #writes a Gslib file and returns a Gslib Instance
        self.filename=filename
        
        if title== 'null':
            self.title=filename
        else: 
            self.title=title
        
        if type(prop_names) is list or type(prop_names) is numpy.ndarray:
            self.prop_names=prop_names
        else:
            self.prop_names=[prop_names]
            
        self.n_properties=len(self.prop_names)
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.data=data
        self.path= f'{folder}/{filename}.out'
        
        #writes the header in the file
        f= open (self.path,'w')
        f.write(f'{self.title}\n')
        f.write(f'{self.n_properties}\n')
        for i in range(self.n_properties):
            f.write(f'{self.prop_names[i]}\n')
        f.close()
        
        #Converst the data to numpy array (only for writing, original data type will be kept in the Gslib instance)

        if type(data) is pandas.core.frame.DataFrame: data=data.to_numpy()
        
        elif type(data) is tuple or type(data) is list: data=numpy.asarray(data)

        #Writes data to file
        f= open (self.path,'a')

        if type(data) is numpy.ndarray:        
            
            if self.n_properties==1:
                if data.shape == (1, self.nx*self.ny*self.nz): data= data.T #if data is a 1D array,it is transposed for writing in a single column
                elif data.shape == (self.nx*self.ny*self.nz,): pass #If data is already in one column
                elif data.shape == (self.nx,self.ny,self.nz): data= numpy.reshape(data, newshape=(nx*ny*nz,1), order='F') #if data is not a 1D array, but a 3D grid, changes shape to the array for writing
                numpy.savetxt(f, data, fmt='%.6f')
            
            elif self.n_properties>1:
                
                if data[0].shape == (1, self.nx*self.ny*self.nz): data= data.T #if is an array of dimension= n_properties
                elif data[0].shape == (self.ny,) or data[0].shape== (self.n_properties,) or data[0].shape == (self.nx*self.ny*self.nz, 1): pass #if data is an array of dimension= n_properties - for well logs
                elif data[0].shape == (self.nx,self.ny,self.nz): 
                    for i in self.n_properties: data[i]= numpy.reshape(data[i], newshape=(nx*ny*nz,1), order='F')

                numpy.savetxt(f, data,fmt='%.6f')
                
            else: return "Cannot read input data correctly"
             
        else: return "Input data type not recognized"
            
        f.close()
        return self
    
    
    
    def Gslib_writethis(self, path):
        #Use this module only when you need to write a Gslib class that has been already defined
        if self.title== 'null':
            self.title=self.filename
        f= open (path,'w')
        f.write(f'{self.title}\n')
        f.write(f'{self.n_properties}\n')
        for i in range(self.n_properties):
            f.write(f'{self.prop_names[i]}\n')
        f.close()
        
        #Converst the data to numpy array (only for writing, original data type will be kept in the Gslib instance)
        if type(self.data) is pandas.core.frame.DataFrame:
            self.data=self.data.to_numpy()
        
        elif type(self.data) is tuple or type(self.data) is  list:
            self.data=numpy.asarray(self.data)

        #Writes data to file
        f= open (path,'a')
        if type(self.data) is numpy.ndarray:        
            
            if self.n_properties==1 and self.data.shape == (1, self.nx*self.ny*self.nz):
                #if data is a 1D array,it is transposed for writing in a single column
                self.data= self.data.T
                numpy.savetxt(f, self.data, fmt='%.6f')
                
            elif self.n_properties==1 and self.data.shape == (self.nx*self.ny*self.nz,):
                #If data is already in one column
                numpy.savetxt(f, self.data, fmt='%.6f')
            
            elif self.n_properties==1 and self.data.shape == (self.nx,self.ny,self.nz):
                #if data is not a 1D array, but a 3D grid, changes shape to the array for writing
                self.data= numpy.reshape(self.data, newshape=(self.nx*self.ny*self.nz,1), order='F')
                numpy.savetxt(f, self.data,fmt='%.6f')
            
            elif self.n_properties>1 and self.data[0].shape == (1, self.nx*self.ny*self.nz): 
                #if is an array of dimension= n_properties
                self.data= self.data.T
                numpy.savetxt(f, self.data,fmt='%.6f')
            
            elif self.n_properties>1 and self.data[0].shape == (self.nx*self.ny*self.nz, 1): 
                #If data is already in n_properties columns
                numpy.savetxt(f, self.data,fmt='%.6f')
            
            elif self.n_properties>1 and self.data[0].shape == (self.nx,self.ny,self.nz): 
                #if input data is n_properties 3D grids, reshape each 3D grid in 1D array
                for i in self.n_properties: 
                    self.data[i]= numpy.reshape(self.data[i], newshape=(self.nx*self.ny*self.nz,1), order='F')
                    
                numpy.savetxt(f, self.data,fmt='%.6f')   
            
            else:
                print ("Cannot read input data correctly")
                return None
             
        else:
            print ("Data type not recognized")
            return None
            
        f.close()
        return None
            
    
    
    def To_sgems(self, write='no', folder='output', magicNumber=1561792946, type_def= "Cgrid" , version= 100, 
                 xsize=1, ysize=1, zsize=1, x0=1, y0=1, z0=1):
        #Creates a Sgems instance from the Gslib instance. When write='yes' it also writes a sgems file in folder
        #1065353216 == 1 in binary file
        try: 
            import Modules.Sgems as sgm
        except:
            import Sgems as sgm         #this import works when this class is launched indipendently (when name==__main__)
            
        sgems_obj= sgm.Sgems()
        
        sgems_obj.filename= self.filename
        
        sgems_obj.title= self.filename
        
        sgems_obj.n_prop= self.n_properties 
        
        if type(self.prop_names) != list:
            self.prop_names= list(self.prop_names)
            
        sgems_obj.property_names= self.prop_names
        
        sgems_obj.nx= self.nx
        
        sgems_obj.ny= self.ny
        
        sgems_obj.nz= self.nz
        
        if type(self.data)!= numpy.ndarray:
            sgems_obj.data= self.data.to_numpy()
        else:
            sgems_obj.data= self.data
            
        sgems_obj.magicNumber    = magicNumber
        sgems_obj.type_def       = type_def
        sgems_obj.version        = version
        
        sgems_obj.xsize          = xsize*1065353216
        sgems_obj.ysize          = ysize*1065353216
        sgems_obj.zsize          = zsize*1065353216
        sgems_obj.x0             = x0*1065353216
        sgems_obj.y0             = y0*1065353216
        sgems_obj.z0             = z0*1065353216
        sgems_obj.n_data        = self.nx*self.ny*self.nz 

        if write=='yes':
            path= f'{folder}/{self.prop_names[0]}.sgems'

            sgems_obj.Sgems_writethis(path)
            print (f'File saved as {self.prop_names[0]}.sgems')
        
        return sgems_obj