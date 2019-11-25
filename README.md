# MobileNetV2

### Link:

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

### Abstract

- New mobile architecture: MobileNetV2. This improves the state of the art performance of mobile models.
- Discusses how to apply these models to the object detection task in the SSD Lite framework
- How to build a mobile semantic segmentation models
- The new model uses some kind of thin bottleneck layers, and a lightweight depthwise convolution on the intermediate expansion. Also, the paper mentions that is important to remove non-linearity in the narrow layers.
- The measurements on performance were made on the databases: ImageNet classification, COCO object detection, VOC image segmentation.
- The trade-offs between accuracy and number of operations measured by multiply-adds, latency and number of parameters were analized


### Introduction

- Problem
    - The modern state of the art networks requires high computational resources beyond the capabilities of many mobile applications.
- The proposed method
    - This paper introduces a new neural network architecture that is specifically tailored for mobile and resource-constrained environments
    - The proposed model significantly decreasing the number of operations and memory needed while retaining the same accuracy.
- Main contributions
    - Novel layer module: The inverted residual with linear bottleneck
    - This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution.

### Related Work

- Methods to improve balance between accuracy and performance
    - Designs of different neural networks
    - Hyperparameter optimization
    - Pruning
    - Connectivity learning
    - Changing the connectivity structure
    - Sparsity
    - Genetic algorithms and reinforcement learning to architectural search
- Drawback is that the resulting networks end up very complex
- Network desing based on MobileNetV1

### Preliminaries, discussion and intuition

- Depthwise Separable Convolutions
    - The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into two separate layers. The first layer is called a depthwise convolution, it performs lightweight filtering by applying a single convolutional filter per input channel.
    - The second layer is a 1X1 convolution, called a pointwise convolution, which is responsible for building new features through computing linear combinations of the input channels.
    - Cost

        Standard convolution takes an hi × wi × di input tensor Li , and applies convolutional kernel K ∈ R k × k × di × dj to produce an hi × wi × d joutput tensor L j . Standard convolutional layers have the computational cost of:

        hi · wi · di · dj · k · k.

        Depthwise separable convolutions are a drop-in replacement for standard convolutional layers. Empirically they work almost as well as regular convolutions but only cost:

        $$h_i . w_i . d_i(k^2 + d_j)$$

        which is the sum of the depthwise and 1 × 1 pointwise convolutions.

    - MobileNetV2 uses k = 3 (3 × 3 depthwise separable convolutions)
- Linear Bottlenecks

    Related with the basic properties of these activation tensors.

    For an input set of real images, we say that the set of layer activations forms a "manifold of interest".

    - Manifolds can be embedded in low-dimensional subspaces
        - In other words, when we look at all individual d-channel pixels of a deep convolutional layer, the information encoded in those values actually lie in some manifold, which in turn is embeddable into a low-dimensional subspace
        - At first glance, such a fact could then be captured and exploited by simply reducing the dimensionality of a layer thus reducing the dimensionality of the operating space.
    - The width multiplier approach allows one to **reduce the dimensionality** of the activation space until the **manifold of interest spans** this entire space.
    - However, this intuition breaks down when we recall that deep convolutional neural networks actually have **non-linear per coordinate transformations,** such as **ReLU**.
        - Deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.

            ![](/images/Untitled-d8cd1d43-b46b-4b72-89ea-7b7b0f8a1bd8.png)

        - When ReLU collapses the channel, it inevitably loses information in that channel. However, if we have lots of channels, and there is a structure in the activation manifold that information might still be preserved in the other channels.
    - **Two properties** that are indicative of the requirement that the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space
        - If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
        - ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.
    - Assuming the manifold of interest is low-dimensional we can capture this by inserting **linear bottleneck layers** into the convolutional blocks.

        ![](/images/Untitled-e1ade487-34a0-4b89-b279-70f71750da0c.png)

        The ratio between the size of the input bottleneck and the inner size as the expansion ratio

- Inverted residuals

    ![](/images/Untitled-a89e96c6-3490-4702-9c32-bfa0130b3fa6.png)

    To improve the ability of a gradient to propagate across multiplier layers shortcuts were added. The inverted design is considerably more memory efficient

- Information flow interpretation
    - There is a separation between
        1. The input/output domains of the building blocks (bottleneck layers) *(**capacity** of the network)*
        2. The layer transformation. That is a non-linear function that converts input to the output (***expressiveness***)
    - When inner layer depth is 0 the underlying convolution is the identity function thanks to the shortcut connection.
    - When the expansion ratio is smaller than 1, this is a classical residual convolutional block
    - However, the expansion ratio greater than 1 is the most useful.

### Model Architecture

> *Basic building block is a bottleneck depth-separable convolution with residuals*

![](/images/Untitled-97b28719-ab36-4e38-8bd5-ab565987a4a2.png)

![](/images/Untitled-c42f7c3e-60fe-41fa-b6c6-e466ecdebbc1.png)

The architecture of MobileNetV2 contains:

- The initial fully convolution layer with 32 filters
- 19 residual bottleneck layers

    ![](/images/Untitled-9b706426-a239-446b-b380-b4f4e816727e.png)

- **Side things:**
    - ReLU6 as the non-linearity because of its robustness when used with low-precision computation
    - Kernel size 3 × 3
    - Dropout and batch normalization during training.
    - With the exception of the first layer, we use a constant expansion rate throughout the network
        - Expansion rates between 5 and 10 result in nearly identical performance curves
    - For all main experiments, the expansion factor was 6 applied to the size of the input tensor
- Trade-off hyper parameters

### Implementation Notes

- Memory efficient inference

    The inverted residual bottleneck layers allow a particularly memory efficient implementation which is very important for mobile applications.

### Experiments

- **Training setup**:
    - Training was made using TensorFlow
    - RMSPropOptimizer with both decay and momentum set to 0.9
    - Batch normalization after every layer
    - The standard weight decay is set to 0.00004.
    - Initial learning rate of 0.045
    - Learning rate decay rate of 0.98 per epoch
    - 16 GPU asynchronous workers
    - Batch size of 96
- **Results**

    ![](/images/Untitled-acdbc4bc-48f9-4c2e-8498-43a2a4e77bc5.png)

    ![](/images/Untitled-97e52e7d-0f41-48fc-8b20-3f2fed38fd87.png)

## Source Code
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py)
