from __future__ import division
import sys
sys.path.insert(0, "./ml")
import os
from optparse import OptionParser
import math
import npNADE as NADE
import Instrumentation
import Backends
import Data
import Training
import numpy as np
import Results
import scipy
import scipy.stats
import h5py
from Utils.MVNormal import MVNormal 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
import matplotlib
import Data.utils

from scipy.interpolate import spline

DESTINATION_PATH = "/home/beni/Dropbox/temp/deep_nade_figures"
vmin = 0 
vmax = 1

def plot_spline():
    pass

def plot_sample(sample, sample_shape, origin='upper'):
    plt.imshow(sample.reshape(sample_shape).copy(), interpolation='nearest', origin=origin, vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def plot_RF(rf, sample_shape):
    norm = matplotlib.colors.Normalize()
    norm.autoscale(rf)
    rf = np.resize(rf, np.prod(sample_shape)).reshape(sample_shape)
    norm_zero = min(max(norm(0.0), 0.0+1e-6), 1.0-1e-6)
    cdict = {
             'red'  :  ((0., 0., 0.), (norm_zero, 0.5, 0.5), (1., 1., 1.)),
             'green':  ((0., 0., 0.), (norm_zero, 0.5, 0.5), (1., 1., 1.)),
             'blue' :  ((0., 0., 0.), (norm_zero, 0.5, 0.5), (1., 1., 1.))
             }
    #generate the colormap with 1024 interpolated values
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)                
    plt.imshow(rf, interpolation='nearest', origin='upper', cmap=my_cmap)   
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False) 

def plot_MNIST_results():
    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(6,4), dpi=100)    
    ll_1hl = [-92.17,-90.69,-89.86,-89.16,-88.61,-88.25,-87.95,-87.71]
    ll_2hl = [-89.17, -87.96, -87.10, -86.41, -85.96, -85.60, -85.28, -85.10] 
    x = np.arange(len(ll_1hl))    
    plt.axhline(y=-84.55, color="black", linestyle="--", label="2hl-DBN")
    plt.axhline(y=-86.34, color="black", linestyle="-.", label="RBM")
    plt.axhline(y=-88.33, color="black", linestyle=":", label="NADE (fixed order)")
    plt.plot(ll_1hl, "r^-", label="1hl-NADE")
    plt.plot(ll_2hl, "go-", label="2hl-NADE")
    plt.xticks(x, 2**x)    
    plt.xlabel("Models averaged")
    plt.ylabel("Test loglikelihood (nats)")
    plt.legend(loc=4, prop = {"size":10})
    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.10)
    plt.savefig(os.path.join(DESTINATION_PATH, "likelihoodvsorderings.pdf"))

def load_model(model_filename):
    print(model_filename)
    try:
        hdf5_route = "/highest_validation_likelihood/parameters"
        params = Results.Results(model_filename).get(hdf5_route)    
    except:
        hdf5_route = "/final_model/parameters"
        params = Results.Results(model_filename).get(hdf5_route)
    model_class = getattr(NADE, params["__class__"])
    return model_class.create_from_params(params)

def plot_examples(nade, dataset, shape, name, rows=5, cols=10):    
    #Show some samples
    images = list()
    for row in xrange(rows):                     
        for i in xrange(cols):
            nade.setup_n_orderings(n=1)
            sample = dataset.sample_data(1)[0].T
            dens = nade.logdensity(sample)
            images.append((sample, dens))
    images.sort(key=lambda x: -x[1])
    
    plt.figure(figsize=(0.5*cols,0.5*rows), dpi=100)
    plt.gray()            
    for row in xrange(rows):                     
        for col in xrange(cols):
            i = row*cols+col
            sample, dens = images[i]
            plt.subplot(rows, cols, i+1)
            plot_sample(np.resize(sample, np.prod(shape)).reshape(shape), shape, origin="upper")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.04, wspace=0.04)
    type_1_font()
    plt.savefig(os.path.join(DESTINATION_PATH, name))                


def plot_samples(nade, shape, name, rows=5, cols=10):    
    #Show some samples
    images = list()
    for row in xrange(rows):                     
        for i in xrange(cols):
            nade.setup_n_orderings(n=1)
            sample = nade.sample(1)[:,0]
            dens = nade.logdensity(sample[:, np.newaxis])
            images.append((sample, dens))
    images.sort(key=lambda x: -x[1])
    
    plt.figure(figsize=(0.5*cols,0.5*rows), dpi=100)
    plt.gray()            
    for row in xrange(rows):                     
        for col in xrange(cols):
            i = row*cols+col
            sample, dens = images[i]
            plt.subplot(rows, cols, i+1)
            plot_sample(np.resize(sample, np.prod(shape)).reshape(shape), shape, origin="upper")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.04, wspace=0.04)
    type_1_font()
    plt.savefig(os.path.join(DESTINATION_PATH, name))                
    #plt.show()

def inpaint_digits_(dataset, shape, model, n_examples = 5, delete_shape = (10,10), n_samples = 5, name = "inpaint_digits"):    
    #Load a few digits from the test dataset (as rows)
    data = dataset.sample_data(1000)[0]
    
    #data = data[[1,12,17,81,88,102],:]
    data = data[range(20,40),:]
    n_examples = data.shape[0]
    
    #Plot it all
    matplotlib.rcParams.update({'font.size': 8})
    plt.figure(figsize=(5,5), dpi=100)
    plt.gray()
    cols = 2 + n_samples
    for row in xrange(n_examples):
        # Original
        plt.subplot(n_examples, cols, row*cols+1)
        plot_sample(data[row,:], shape, origin="upper")        
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.40, wspace=0.04)
    plt.savefig(os.path.join(DESTINATION_PATH, "kk.pdf"))
    

def inpaint_digits(dataset, shape, model, n_examples = 5, delete_shape = (10,10), n_samples = 5, name = "inpaint_digits"):    
    #Load a few digits from the test dataset (as rows)
    data = dataset.sample_data(1000)[0]
    
    #data = data[[1,12,17,81,88,102],:]    
    data = data[[1,12,17,81,88,37],:]
    n_examples = data.shape[0]
    
    #Generate a random region to delete
    regions = [ (np.random.randint(shape[0]-delete_shape[0]+1), np.random.randint(shape[1]-delete_shape[1]+1)) for i in xrange(n_examples)]
    print(regions)
    regions = [(11,5), (11,5), (11,5), (4,13), (4,13), (4,13)]
    
    #Generate masks
    def create_mask(x,y):
        mask = np.ones(shape)
        mask[y:y+delete_shape[1], x:x+delete_shape[0]] = 0
        return mask.flatten()
    masks = [create_mask(x,y) for (x,y) in regions]    
    #Hollow
    def hollow(example, mask):
        hollowed = example.copy()
        return hollowed * mask
    hollowed = [hollow(data[i,:], mask) for i,mask in enumerate(masks)] 
    
    densities = model.logdensity(data.T)
    #Calculate the marginal probability under a nade
    marginal_densities = [model.marginal_density(h, mask) for h, mask in zip(hollowed, masks)]

    #Generate some samples    
    samples = [model.sample_conditional(h, mask, n_samples=n_samples) for h, mask in zip(hollowed, masks)]
    #samples = [model.sample_conditional_max(h, mask, n_samples=n_samples) for h, mask in zip(hollowed, masks)]
    
    #Plot it all
    matplotlib.rcParams.update({'font.size': 8})
    plt.figure(figsize=(5,5), dpi=100)
    plt.gray()
    cols = 2 + n_samples
    for row in xrange(n_examples):
        # Original
        plt.subplot(n_examples, cols, row*cols+1)
        plot_sample(data[row,:], shape, origin="upper")
        plt.title("%.2f" % densities[row])
        # Marginalization region
        plt.subplot(n_examples, cols, row*cols+2)
        plot_sample(hollowed[row], shape, origin="upper")
        plt.gca().add_patch(plt.Rectangle(regions[row], delete_shape[0], delete_shape[1], facecolor="red", edgecolor="red"))        
        plt.title("%.2f" % marginal_densities[row])        
        # Samples
        for j in xrange(n_samples):
            plt.subplot(n_examples, cols, row*cols+3+j)
            plot_sample(samples[row][:,j], shape, origin="upper")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.40, wspace=0.04)
    plt.savefig(os.path.join(DESTINATION_PATH, name+".pdf"))
    
    
def plot_RF_of_ps(rf, sample_shape):
    rf = np.resize(rf, np.prod(sample_shape)).reshape(sample_shape)
    plt.imshow(rf, interpolation='nearest', origin='upper', vmin=0, vmax=1.0)   
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False) 

def plot_RFs(name, nade, sample_shape, rows=5, cols=10):
    rf_sizes = []
    W = nade.W1
    for i in xrange(W.shape[1]):
        rf_sizes.append((i, -(W[:,i]**2).sum()))
    rf_sizes.sort(key = lambda x: x[1])
    plt.figure(figsize=(0.5*cols,0.5*rows), dpi=100)    
    plt.gray()
    for i in xrange(rows):
        for j in xrange(cols):
            n = i*cols+j                
            plt.subplot(rows,cols, n+1)
            rf = nade.W1[:,rf_sizes[n][0]]
            #plot_RF(rf, sample_shape)
            plot_RF_of_ps(np.exp(rf)/(1+np.exp(rf)), sample_shape)
    plt.tight_layout(0.2, 0.2, 0.2)
    #plt.savefig(os.path.join(DESTINATION_PATH, name + "_W1.pdf"))    
    plt.figure(figsize=(0.5*cols,0.5*rows), dpi=100)
    plt.gray()
    for i in xrange(rows):
        for j in xrange(cols):
            n = i*cols+j
            plt.subplot(rows,cols, n+1)
            rf = np.resize(nade.Wflags[:,rf_sizes[n][0]], np.prod(sample_shape)).reshape(sample_shape)
            plot_RF(rf, sample_shape)    
            #plot_RF_of_ps(np.exp(rf)/(1+np.exp(rf)), sample_shape)
    plt.tight_layout(0.2, 0.2, 0.2)
    #plt.savefig(os.path.join(DESTINATION_PATH, name + "_Wflags.pdf"))
    
    plt.figure(figsize=(0.5*cols,0.5*rows), dpi=100)
    plt.gray()
    for i in xrange(rows):
        for j in xrange(cols):
            n = i*cols+j
            plt.subplot(rows,cols, n+1)
            w  = nade.W1[:,rf_sizes[n][0]]
            f = nade.Wflags[:,rf_sizes[n][0]]
            
            x = np.log( (1-np.exp(f-1))/(1+np.exp(w+f)) )
            y = w - x

            rf = np.resize(x, np.prod(sample_shape)).reshape(sample_shape)
            #plot_RF_of_ps(rf, sample_shape)
            plot_RF(rf, sample_shape)    
    plt.tight_layout(0.2, 0.2, 0.2)    


    plt.figure(figsize=(0.5*cols,0.5*rows), dpi=100)
    plt.gray()
    for i in xrange(rows):
        for j in xrange(cols):
            n = i*cols+j
            plt.subplot(rows,cols, n+1)
            w  = nade.W1[:,rf_sizes[n][0]]
            f = nade.Wflags[:,rf_sizes[n][0]]
            
            x = np.log( (1-np.exp(f-1))/(1+np.exp(w+f)) )
            y = w - x

            rf = np.resize(y, np.prod(sample_shape)).reshape(sample_shape)
            #plot_RF_of_ps(rf, sample_shape)
            plot_RF(rf, sample_shape)    
    plt.tight_layout(0.2, 0.2, 0.2)    
    
    plt.show()


def main():
    global vmin, vmax
    mnist = True
    bsds = False
    if mnist:
        dataset_file = os.path.join(os.environ["DATASETSPATH"], "original_NADE/binarized_mnist.hdf5" )
        test_dataset = Data.BigDataset(dataset_file,  "test", "data")    
        nade2hl = load_model(os.path.join(os.environ["RESULTSPATH"], "orderless/mnist/2hl/NADE.hdf5"))        
        #plot_MNIST_results()
        #exit()                
        #plot_examples(nade2hl, test_dataset, (28,28), "mnist_examples.pdf", rows = 5, cols=10)        
        #plot_samples(nade2hl, (28,28), "mnist_samples_2hl.pdf", rows = 5, cols=10)        
        plot_RFs("MNIST_2hl", nade2hl, sample_shape = (28,28))                
        #nade2hl = load_model(os.path.join(os.environ["RESULTSPATH"], "orderless/mnist/2hl/NADE.hdf5"))        
        #np.random.seed(1) #1,17,43,49
        #nade2hl.setup_n_orderings(10)        
        #inpaint_digits(test_dataset, (28,28), nade2hl, delete_shape=(10,10), n_examples = 6, n_samples = 5)    

    if bsds:
        vmin = -0.5 
        vmax = 0.5
        np.random.seed(1)
        dataset_file = os.path.join(os.environ["DATASETSPATH"], "natural_images/BSDS300/BSDS300_63_no_DC_val.hdf5" )
        test_dataset = Data.BigDataset(dataset_file,  "test/.", "patches")
        nade = load_model(os.path.join(os.environ["RESULTSPATH"], "orderless/BSDS300/6hl/NADE.hdf5"))
        plot_examples(nade, test_dataset, (8,8), "BSDS_data.pdf", rows = 5, cols=10)        
        plot_samples(nade, (8,8), "BSDS_6hl_samples.pdf", rows = 5, cols=10)
        nade = load_model(os.path.join(os.environ["RESULTSPATH"], "orderless/BSDS300/2hl/NADE.hdf5"))
        plot_RFs("BSDS_2hl", nade, sample_shape = (8,8))
    exit()
    
    np.random.seed(8341)
    show_RFs = False
    print_likelihoods = False
    print_mixture_likelihoods = True 
    show_data = False
    show_samples = False
    denoise_samples = False
    n_orderings = 6
    n_samples = 10
    #sample_shape = (28, 28)
    sample_shape = (8, 8)
        
    parser = OptionParser(usage = "usage: %prog [options] dataset_file nade_path")
    parser.add_option("--training_samples", dest = "training_route", default="train")
    parser.add_option("--validation_samples", dest = "validation_route", default="validation")
    parser.add_option("--test_samples", dest = "test_route", default="test")
    parser.add_option("--samples_name", dest = "samples_name", default="data")
    parser.add_option("--normalize", dest = "normalize", default=False, action="store_true")    
    parser.add_option("--add_dimension", dest = "add_dimension", default=False, action="store_true")
    (options, args) =  parser.parse_args()

    dataset_filename = os.path.join(os.environ["DATASETSPATH"], args[0])
    model_filename = os.path.join(os.environ["RESULTSPATH"], args[1])
    print(model_filename)
    try:
        hdf5_route = "/highest_validation_likelihood/parameters"
        params = Results.Results(model_filename).get(hdf5_route)    
    except:
        hdf5_route = "/final_model/parameters"
        params = Results.Results(model_filename).get(hdf5_route)
    model_class = getattr(NADE, params["__class__"])
    nade = model_class.create_from_params(params)
    #Load datasets
    print("Loading datasets")
    dataset_file = os.path.join(os.environ["DATASETSPATH"], dataset_filename)
    training_dataset = Data.BigDataset(dataset_file, options.training_route, options.samples_name)
    validation_dataset = Data.BigDataset(dataset_file, options.validation_route, options.samples_name)
    test_dataset = Data.BigDataset(dataset_file,  options.test_route, options.samples_name)        
    n_visible = training_dataset.get_dimensionality(0)
    
    if options.normalize:
        mean, std = Data.utils.get_dataset_statistics(training_dataset)
        training_dataset = Data.utils.normalise_dataset(training_dataset, mean, std)            
        validation_dataset = Data.utils.normalise_dataset(validation_dataset, mean, std)                
        test_dataset = Data.utils.normalise_dataset(test_dataset, mean, std)    
     
    #Setup a list with n_orderings orderings
    print("Creating random orderings")
    orderings = list()    
    #orderings.append(range(nade.n_visible))
    for i in xrange(n_orderings):
        o = range(nade.n_visible)
        np.random.shuffle(o)
        orderings.append(o)
    
    #Print average loglikelihood and se for several orderings
    if print_likelihoods:
        nade.setup_n_orderings(orderings = orderings)
        ll = nade.get_average_loglikelihood_for_dataset(test_dataset)
        print("Mean test-loglikelihood (%d orderings): %.2f" % (orderings, ll))
    
    if print_mixture_likelihoods:
        #d = test_dataset.sample_data(1000)[0].T
        d = test_dataset.get_data()[0].T
        for n in [1,2,4,8,16,32,64,128]:
            #nade.setup_n_orderings(n)                     
            multi_ord = nade.logdensity(d)    
            print(n, np.mean(multi_ord), scipy.stats.sem(multi_ord))    

    if show_RFs:
        rf_sizes = []
        W = nade.W1.get_value()
        for i in xrange(W.shape[1]):
            rf_sizes.append((i, -(W[:,i]**2).sum()))
        rf_sizes.sort(key = lambda x: x[1])
        plt.figure()
        plt.gray()
        for i in xrange(10):
            for j in xrange(10):
                n = i*10+j                
                plt.subplot(10,10, n+1)
                rf = nade.Wflags.get_value()[:,rf_sizes[n][0]]
                plot_RF(rf, sample_shape)
        plt.figure()
        plt.gray()
        for i in xrange(10):
            for j in xrange(10):
                n = i*10+j
                plt.subplot(10,10, n+1)
                rf = np.resize(nade.W1.get_value()[:,rf_sizes[n][0]], np.prod(sample_shape)).reshape(sample_shape)
                plot_RF(rf, sample_shape)
        plt.show()
        
    #Show some samples
    if show_samples:
        images = []       
        for row,o in enumerate(orderings):                     
            samples = nade.sample(n_samples)
            for i in xrange(samples.shape[1]):
                nade.setup_n_orderings(n=1)
                sample = samples[:,i]                
                dens = nade.logdensity(sample[:, np.newaxis])
                if options.add_dimension:                    
                    sample_extra_dim = np.resize(sample, len(sample)+1)
                    sample_extra_dim[-1] = -sample.sum()
                    images.append((sample_extra_dim, dens))                    
                else:
                    images.append((sample, dens))                                    
        images.sort(key=lambda x: -x[1])
        plt.figure()
        plt.gray()        
        for row,o in enumerate(orderings):                     
            for i in xrange(samples.shape[1]):
                plt.subplot(n_orderings, n_samples, row*n_samples+i+1)
                im_ll = images[row*n_samples+i]
                plot_sample(im_ll[0], sample_shape)                
                plt.title("%.1f" % im_ll[1], fontsize=9)                
        plt.show()

    #Show some data
    if show_data:
        images = []
        for row,o in enumerate(orderings):                     
            samples = test_dataset.sample_data(n_samples)[0].T
            for i in xrange(samples.shape[1]):
                nade.setup_n_orderings(n=1)                
                sample = samples[:,i]
                dens = nade.logdensity(sample[:, np.newaxis])
                if options.add_dimension:                    
                    sample_extra_dim = np.resize(sample, len(sample)+1)
                    sample_extra_dim[-1] = -sample.sum() 
                    images.append((sample_extra_dim, dens))                    
                else:
                    images.append((sample, dens))
        images.sort(key=lambda x: -x[1])
        plt.figure()
        plt.gray()        
        for row,o in enumerate(orderings):                     
            for i in xrange(samples.shape[1]):
                plt.subplot(n_orderings, n_samples, row*n_samples+i+1)
                im_ll = images[row*n_samples+i]
                plot_sample(im_ll[0], sample_shape)                
                plt.title("%.1f" % im_ll[1], fontsize=9)                
        plt.show()

    #Get a sample and clean it by taking pixels randomly and assigning the most probably value given all the others
    if denoise_samples:
        n = 10
        nade.set_ordering(orderings[-1])
        sample = nade.sample(n)
        plt.figure()
        plt.gray()
        for it in xrange(10):        
            for s in xrange(n):
                logdensities = nade.logdensity(sample)
                plt.subplot(n,10, s*10+it+1)
                plot_sample(sample[:,s], sample_shape)            
                plt.title("%.1f" % logdensities[s])
            #i = np.random.randint(sample.shape[0])
            for i in xrange(sample.shape[0]):
                j = np.random.randint(sample.shape[0])
                #mask = sample            
                mask = np.ones_like(sample)        
                mask[j] = 0
                sample[j] = 0        
                ps = nade.conditional(sample, mask)
                #sample[j] = ps[j] > np.random.rand()        
                sample[j] = ps[j] > 0.5
        plt.show()

main()