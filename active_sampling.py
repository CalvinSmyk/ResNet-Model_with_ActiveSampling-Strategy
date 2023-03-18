import numpy as np

class Similarity_measures():
    def __init__(self,first_img,snd_img):
        self.first_img = first_img
        self.snd_img = snd_img
        self.max_pixel = 32*32 #Max Image is 64*64 pixels
        self.MSE = self.Mean_squared_error()
    """Input needs to be two np arrays with 
        shape [rows,colums,channels]"""

    def Mean_squared_error(self):
        mse = []
        for i in range(self.first_img.shape[2]):
            diff = np.subtract(self.first_img[:,:,i],self.snd_img[:,:,i])
            mean = np.mean(np.square(diff / self.max_pixel))
            mse.append(mean)
        return mse

    def root_mean_squared_error(self):
        rmse_bands = []

        for i in self.MSE:
            std = np.sqrt(i)
            rmse_bands.append(std)

        return np.mean(rmse_bands)

    def peek_signal_to_noise_ratio(self):
        return 20*np.log10(self.max_pixel) - 10.0 * np.log10(np.mean(self.MSE))

    def similarity_measurement(self):
        constant = 1e-10
        numerator = 2*self.first_img*self.snd_img + constant
        denominator = np.power(self.first_img,2) + np.power(self.snd_img,2) + constant
        return (numerator / denominator)

    def Entropy_Histogram_Similarity(self):
        H = (np.histogram2d(self.first_img.flatten(),self.snd_img.flatten()))[0]
        return -np.sum(np.nan_to_num(H*np.log2(H)))

    def Information_theoretic_based_statistic_Similarity(self):
        pass

    def sre(self):
        """
        Signal to Reconstruction Error Ratio
        """

        org_img = self.first_img.astype(np.float32)

        sre_final = []
        for i in range(org_img.shape[2]):
            numerator = np.square(np.mean(org_img[:, :, i]))
            denominator = (np.linalg.norm(org_img[:, :, i] - self.snd_img[:, :, i])) / (
                    org_img.shape[0] * org_img.shape[1]
            )
            sre_final.append(numerator / denominator)

        return 10 * np.log10(np.mean(sre_final))
