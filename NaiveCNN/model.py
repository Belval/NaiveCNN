import numpy as np

from .utils import im2col

class ConvolutionalNet():
    """
        Description:
            The definition of our naive CNN, it is made of
                One convolution layer
                One ReLU layer
                One pooling layer
                One fully connected layer
                One softmax layer / output layer
            It is loosely based on the ninth chapter of the excellent deep learning book
            (http://www.deeplearningbook.org/contents/convnets.html)
        Good read on the subject:
            Deep Learning Book (http://www.deeplearningbook.org/)
            Understanding the difficulty of training deep feedforward neural neural network (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
            Delving Deep into Rectifiers (https://arxiv.org/pdf/1502.01852.pdf)
    """

    def __init__(self, kernel_count, kernel_size, input_size, output_size):
        """
            Description: The ConvNet contructor.
            Parameters:
                kernel_count -> The number of kernel to use
                kernel_size -> The size of the kernel to use (always a square)
                output_size -> The number of classes
                input_size -> A tuple that defines the size of the input
        """
        
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.output_size = output_size

        # Random initialization of our kernels based on a gaussian distribution with a std of 1.0
        # See https://arxiv.org/pdf/1502.01852.pdf Page 3
        self.kernels = [
            np.random.normal(size=(kernel_size, kernel_size)) for x in range(kernel_count)
        ]

    @staticmethod
    def _convolution(inputs, kernels):
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

        inp_sp = np.shape(inputs)
        # We reshape it so every filter is considered an image.
        # For example, 10 x 4 x 32 x 32 -> 40 x 1 x 32 x 32
        reshaped = np.reshape(inputs, (inp_sp[0] * inp_sp[1], 1, inp_sp[2], inp_sp[3]))
        # This will reshape the previously reshaped input to (if the size is 2) 4 x 10240
        tile_col = im2col(reshaped, size, stride=stride, padding=padding)
        # We take the max of each column
        max_ids = np.argmax(tile_col, axis=0)
        # We get the resulting 1 x 10240 vector
        result = tile_col[max_ids, range(max_ids.size)]

        new_size = (inp_sp[2] - size + 2 * padding) / stride + 1

        result = np.reshape(result, (new_size, new_size, inp_sp[0], inp_sp[1]))

        # Make it from 16 x 16 x 10 x 4 to 10 x 4 x 16 x 16
        return np.transpose(result, (2, 3, 0, 1))

    @staticmethod
    def _avg_pool(inputs, size, stride, padding):
        """
            (Copy & paste of the max pool code with np.mean instead of np.argmax)
            Description: Average pool layer
            Parameters:
                inputs -> The input of size [batch_size] x [filter] x [shape_x] x [shape_y]
                size -> The size of the tiling
                stride -> The applied translation at each step
                padding -> The padding (padding with 0 so the last column isn't left out)
        """

        inp_sp = np.shape(inputs)
        reshaped = np.reshape(inputs, (inp_sp[0] * inp_sp[1], 1, inp_sp[2], inp_sp[3]))
        tile_col = im2col(reshaped, size, stride=stride, padding=padding)
        max_ids = np.mean(tile_col, axis=0)
        result = tile_col[max_ids, range(max_ids.size)]
        new_size = (inp_sp[2] - size + 2 * padding) / stride + 1
        result = np.reshape(result, (new_size, new_size, inp_sp[0], inp_sp[1]))
        return np.transpose(result, (2, 3, 0, 1))

    @staticmethod
    def _rectified_linear(inputs):
        """
            Description: Rectified Linear Unit layer (ReLU)
        """

        return np.maximum(inputs, 0, inputs)

    @staticmethod
    def _fully_connected(inputs, weights, unit_count):
        """
            Description: Fully connected layer
            Parameters:
                unit_count -> The number of units in the layer
        """

        return np.dot(inputs, np.reshape(weights, (np.shape(inputs), np.shape(unit_count))))

    @staticmethod
    def _softmax(inputs):
        """
            Description: Softmax function for the output layer
        """

        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forwardpropagation(self, inputs):
        """
            Description: Gives a response based on input
        """

        # My goal was to do something like this, but it's unreadable
        #return _fully_connected(_max_pooling(_rectified_linear(_convolution(inputs, kernels))))

        res_conv = _convolution(inputs, self.kernels)
        res_relu = _rectified_linear(res_conv)
        res_pool = _avg_pool(res_relu)
        res_full = _fully_connected(res_pool, self.full_connected_weights, self.unit_count)
        return _softmax(res_full)

    def backpropagation(self, mean_squared_error):
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

        # For the sake of simplicity we use Mean Squared Error
        for x in range(iteration_count):
            print('Iteration #{}'.format(x))
            errors = np.zeros((batch_size, self.output_size))
            for y in range(batch_size):
                errors[y, :] = (self.feedforward(data[x * batch_size + y]) - labels[x * batch_size + y])**2
            self.backpropagation(np.mean(errors, axis=1))

    def test(self, data, labels):
        """
            Description: Test the ConvNet
            Parameters:
                data -> The data to be used for testing
                labels -> The labels to be used for testing
        """

        good = 0
        for x in range(np.shape(data)[0]):
            if np.argmax(feedforward(data[x, :])) == np.argmax(labels[x, :])
                good += 1

        print('The network successfully identified {} / {} examples.'.format(good, np.shape(data)[0]))