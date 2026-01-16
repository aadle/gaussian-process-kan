import gpjax as gpx

def init_gp(X, y):
    D = gpx.Dataset(X, y)
    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

def main():
    pass

if __name__ == "__main__":
    main()
