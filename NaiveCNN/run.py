import sys
import numpy

from model import ConvolutionalNet
from data_manager import load_data


def main(args):
    """
        Description: Create, train and test the conv net
    """

    # Get the data
    train_data, train_labels, test_data, test_labels = load_data(args[1])

    # Build our ConvNet
    conv_net = ConvolutionalNet(
        args[5], 
        args[6], 
        np.shape(train_data[0])[1], 
        np.shape(test_data[0])[1]
    )

    # Train the ConvNet
    conv_net.train(train_data, train_labels, args[2], args[3], args[4])

    # Test the ConvNet
    conv_net.test(test_data, test_labels)

    # Usage: run.py [data_path] [batch_size] [iteration_count] [alpha] [kernel_count] [kernel_size]
    main(sys.argv)
