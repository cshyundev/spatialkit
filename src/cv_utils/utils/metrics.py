# from ..hybrid_operations import *


# def compute_mse(x:ArrayLike, y:Array, mask:Optional[Array]=None):
#     assert (x.shape == y.shape),(f"Shape of x and y must be same, but x:{str(x.shape)}, y:{str(y.shape)}")
#     if mask:
#         x = x[mask,:]
#         y = y[mask,:]
#     return ((x-y)**2).mean()

# def compute_psnr(x:Array, y:Array,mask:Optional[Array]=None):
#     mse = compute_mse(x,y,mask)
#     return -10.0 * np.log10(mse.item())

# def compute_ssim(x:Array, y:Array,mask:Optional[Array]=None):
#     if mask:
#         x = x[mask,:]
#         y = y[mask,:]
    
#     x = convert_numpy(x)
#     y = convert_numpy(y)
#     # SSIM
#     k1 = 0.01
#     k2 = 0.03
    
#     mu1 = x.mean()
#     mu2 = y.mean()
    
#     var1 = np.var(x)
#     var2 = np.var(y) 
     
#     cov = np.cov(x.flatten(), y.flatten())[0][1]
    
#     c1 = k1 ** 2
#     c2 = k2 ** 2
    
#     numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
#     denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
#     ssim = numerator / denominator
    
#     return ssim
