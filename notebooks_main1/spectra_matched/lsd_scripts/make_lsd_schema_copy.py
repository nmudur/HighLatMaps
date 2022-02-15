from astropy.table import Table
import os
import numpy as np

def create_LSD_schema(schema_filename,t,primary_key,ra_key,dec_key, primary_key_is_column=True):
    """
    schema_filename: The name of the schema file you want to create
    t: the astropy.table.Table object
    primary_key: the string corresponding to the primary key column (I am assuming it is already a column in the fits table)
    ra_key: the string corresponding to the ra column (used for cross-matching)
    dec_key: the string corresponding to the dec column (used for cross-matching)
    primary_key_is_column: a boolean indicating whether the primary_key you've passed is already a column in the table; otherwise create one
    """
    file=open(schema_filename,"w")
    file.write("filters : { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False } \n")
    file.write("schema: \n")
    file.write("  common: \n")
    file.write("    primary_key: {} \n".format(primary_key))
    file.write("    spatial_keys: [{}, {}] \n".format(ra_key,dec_key))
    file.write("    columns: \n")

    for i in range(0,len(t[1].columns)):

        coltype=t[1].columns[i].dtype.kind
        colname=t[1].columns[i].name  #.lower() #commented out for dr17
        colsize=t[1].columns[i].dtype.itemsize
            
        if (coltype=="U") or (coltype=="S"):
            coltype="a"
                
        if (colname==primary_key):
            coltype='u'
            colsize=8
            
        #if column is not a vector...
        if len(t.columns[i].data.shape)==1:
            
            file.write("    - [{}, {}{}] \n".format(colname,coltype,colsize))
    
        #if column is a vector...
        else:
            vectorlength=t.columns[i].data.shape[1]
            file.write("    - [{}, {}{}{}] \n".format(colname,vectorlength,coltype,colsize))
           
    #add primary key column if it doesn't already exist in the table you passed 
    if primary_key_is_column==False:
        file.write("    - [{}, {}{}] \n".format(primary_key,"u",8))
        
    file.close()
    

#t=Table.read("/n/fink2/www/czucker/G314/diagnostics/compiled/G314_diagnostics.fits",format='fits')
#create_LSD_schema("G314.yaml",t,"id","ra","dec",primary_key_is_column=False)
if __name__=='__main__':
    t=Table.read("/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/catalogs/specObj-dr17.fits",format='fits')
    create_LSD_schema(os.environ['LSD_DB']+"sdss_dr17_specobj_caps.yaml",t,"lsd_id","PLUG_RA","PLUG_DEC",primary_key_is_column=False)
