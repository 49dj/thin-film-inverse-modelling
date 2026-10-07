import matplotlib.pyplot as plt

def plot_dual_angle(sigma,datasets,curves,path):
    fig,axes=plt.subplots(2,len(datasets),figsize=(10,6),squeeze=False)
    for i,((angle,y),fit) in enumerate(zip(datasets,curves)):
        axes[0,i].plot(sigma,y,label='Synthetic observation')
        axes[0,i].plot(sigma,fit,'--',label='Fit')
        axes[0,i].set(title=f'{angle:g} degrees',ylabel='Reflectance')
        axes[0,i].legend()
        axes[1,i].plot(sigma,fit-y)
        axes[1,i].axhline(0,color='black',linewidth=0.7)
        axes[1,i].set(xlabel='Wavenumber (cm^-1)',ylabel='Residual')
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
