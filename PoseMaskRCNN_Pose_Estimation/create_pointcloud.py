import numpy as np

def perspectiveDepthImageToPointCloud(image_depth,defaultValue,perspectiveAngle,clip_start,clip_end,resolutionX,resolutionY,resolution_big,pixelOffset_X_KoSyTopLeft,pixelOffset_Y_KoSyTopLeft):
    '''
        Input: Depth image in perspective projection
        Output: Point cloud as list (in meter)
		
		Parameter:
        - image_depth: Depth image in perspective projection with shape (resolutionY,resolutionX,1)
        - defaultValue: Default value to indicate missing depth information in the depth image
        - perspectiveAngle: Perspective angle in deg
		- clip_start: Near clipping plane in meter
		- clip_end: Far clipping plane in meter
        - resolutionX: resolutionX of the input image
        - resolutionY: resolutionY of the input image
		- resolution_big: resolution_big of the input image
        - pixelOffset_X_KoSyTopLeft: Offset in x direction in pixel from coordinate system top left
        - pixelOffset_Y_KoSyTopLeft: Offset in y direction in pixel from coordinate system top left
    '''
    
    assert(image_depth.shape==(resolutionY,resolutionX,1))
    # Warning: Point cloud will not be correct when depth image was resized!
    
    image_big=np.zeros((resolution_big,resolution_big))
    image_big[pixelOffset_Y_KoSyTopLeft:pixelOffset_Y_KoSyTopLeft+resolutionY,pixelOffset_X_KoSyTopLeft:pixelOffset_X_KoSyTopLeft+resolutionX]=image_depth[:,:,0]
    image_depth=image_big
    image_depth=np.rot90(image_depth,k=2,axes=(0,1))
    
    point_cloud=[]
    range_=clip_end-clip_start
    
	# Loop over all pixels in the depth image:
    for j in range(image_depth.shape[0]):
        for i in range(image_depth.shape[1]):
            if image_depth[j,i]==defaultValue:
                continue
            
            world_z=(image_depth[j,i]*range_+clip_start)
            # Calculate the orthogonal size based on current depth (function of z value)
            orthoSizeZ_x=np.tan(np.deg2rad(perspectiveAngle/2))*world_z*2*resolutionX/resolution_big
            orthoSizeZ_y=np.tan(np.deg2rad(perspectiveAngle/2))*world_z*2*resolutionY/resolution_big
            
            meterPerPixel_x=orthoSizeZ_x/resolutionX
            meterPerPixel_y=orthoSizeZ_y/resolutionY
            
            world_x=(i+0.5-resolution_big/2)*meterPerPixel_x
            world_y=(j+0.5-resolution_big/2)*meterPerPixel_y
            
            p=[world_x,world_y,world_z]
            point_cloud.append(p)
            print(point_cloud)
    return point_cloud

