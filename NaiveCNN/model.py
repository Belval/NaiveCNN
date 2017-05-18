import numpy as np

class ConvolutionalNet():
    """
        Description:
            The definition of our naive CNN, it is made of
                One convolution layer
                One detector layer
                One pooling layer
                One fully connected layer
                One softmax layer / output layer
            It is loosely based on the ninth chapter of the excellent deep learning book
            (http://www.deeplearningbook.org/contents/convnets.html)
    """

    def __init__(self, kernel_count, kernel_size, input_size, output_size):
        """
            Description: The ConvNet contructor.
            Parameters:
                kernel_count -> The number of kernel to use
                kernel_size -> The size of the kernel to use (always a square)
                output_count -> The number of classes
                input_size -> A tuple that defines the size of the input
        """

    @staticmethod
    def _convolution(inputs):
        """
            Description: Convolution layer
        """

    @staticmethod
    def _max_pool(inputs, size, stride, padding):
        """
            Description: Max pool layer
            Parameters:
                inputs -> The input of size [batch_size] x [filter] x [shape_x] x [shape_y]
                size -> The size of the tiling
                stride -> The applied translation at each step
                padding -> The padding (padding with 0 so the last column isn't left out)
        """

        if padding >= stride:
            padding = size 

        inp_sp = np.shape(inputs)
        # We reshape it so every filter is considered an image.
        # For example, 10 x 4 x 32 x 32 -> 40 x 1 x 32 x 32
        reshaped = np.reshape(inputs, (inp_sp[0] * inp_sp[1], 1, inp_sp[2], inp_sp[3]))
        # This will reshape the previously reshaped input to (if the size is 2) 4 x 10240
        tile_col = im2col_indices(reshaped, size, size, stride=stride, padding=padding)
        # We take the max of each column
        max_ids = np.argmax(tile_col, axis=0)
        # We get the resulting 1 x 10240 vector
        result = tile_col[max_ids, range(max_ids.size)]
        # Reshape to get matrices
        # 2,2,0 -> 28 / 2 = 14
        # 2,2,1 -> 28 / 2 = 14
        # 2,3,0 -> 28 / 3 = 9
        # 2,3,1 -> 29 / 3 = 9
        # 2,3,1 -> 30 / 3 = 10
        result = np.reshape(result, )

        # Make it from 16 x 16 x 10 x 4 to 10 x 4 x 16 x 16 
        return np.reshape(result, (2, 3, 0, 1))

    @staticmethod
    def _avg_pool(inputs, size, string, padding):
        """
            Description: Average pool layer
            Parameters:
                inputs -> The input of size [batch_size] x [filter] x [shape_x] x [shape_y]
                size -> The size of the tiling
                stride -> The applied translation at each step
                padding -> The padding (padding with 0 so the last column isn't left out)
        """

    @staticmethod
    def _rectified_linear(inputs):
        """
            Description: Rectified Linear Unit layer (ReLU)
        """

    @staticmethod
    def _fully_connected(input, weights, unit_count):
        """
            Description: Fully connected layer
            Parameters:
                unit_count -> The number of units in the layer
        """

    @staticmethod
    def _softmax(input, weights, output):
        """
            Description: Softmax function for the output layer
        """

    @staticmethod
    def _backpropagation():
        """
            Description: Weight adjusting algorithm
        """

    def train(self, data, labels, batch_size, iteration_count, alpha):
        """
            Description: Train the ConvNet
            Parameters:
                data -> The data to be used for training
                labels -> The labels for the data
                batch_size -> The size of a batch used for one iteration
                iteration_count -> The number of iterations for a full training
                alpha -> The learning rate alpha
        """

    def test(self, data, labels):
        """
            Description: Test the ConvNet
            Parameters:
                data -> The data to be used for testing
                labels -> The labels to be used for testing
        """
