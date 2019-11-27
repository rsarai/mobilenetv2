# Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution


**Authors:**:
Yunpeng Chen †‡ , Haoqi Fan † , Bing Xu † , Zhicheng Yan † , Yannis Kalantidis † , Marcus Rohrbach † , Shuicheng Yan ‡♭ , Jiashi Feng ‡ † Facebook AI, ‡ National University of Singapore, ♭ Yitu Technology

### Abstract

- The output feature maps of a convolution layer can also be seen as a mixture of information at different frequencies.
- The propose is to factorize the mixed feature maps by their frequences and design a novel Octave Convolution operation to store and process feature maps that vary spatially slower” at a lower spatial resolution reducing both memory and computation cost.
    - Single layer
    - Generic
    - Plug-and-play unit
    - To be replace vanilla convolutions wihout adjustments
- The results showed that replacing convolutions with OctConv bosted accuracy while reducing memory and computational cost.
- An OctConv-equipped ResNet-152 can achieve 82.9% top-1 classification accuracy on ImageNet with merely 22.2 GFLOPs

### Introduction

- A natural image can be decomposed into a low spatial frequency component that describes
the smoothly changing structure and a high spatial frequency component that describes the rapidly changing fine details

![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled.png)

- The paper discusses that the output of feature maps of a convolution layer can also be decomposed into features of different spatial frequencies and propose a novel multi-frequency feature representation
- Generalized the vanilla convolution and proposed Octave Convolution (OctConv) which takes in feature maps containing tensors of two frequencies one octave apart, and extracts information directly from the low-frequency maps without the need of decoding it back
- Their contributions
    1. Factorize convolutional feature maps into two groups at different spatial frequencies and process them with different convolutions at their corresponding frequency, one octave apart. As the resolution for low frequency maps can be reduced, this saves both storage and computation. This also helps each layer gain a larger receptive field to capture more contextual information
    2. Plug-and-play operation named OctConv to replace the vanilla convolution for operating on the new feature representation directly and reducing spatial redundancy. Importantly, OctConv is fast in practice and achieves a speedup close to the theoretical limit.
    3. An extensive study of the properties of the proposed OctConv on a variety of backbone CNNs for image and video tasks with significant performance gain.

### Related Work

- Multiple researches improving efficiency of CNN
- Multi-scale Representation Learning: used before deep learning for local feature extraction (its strong robustness and generalization ability)

### Method

- Octave Feature Representation
    - PROBLEM: In Vanilla Convolution all input and output feature maps have the same spatial resolution, which may not be necessary (redundancy)
    - SOLUTION:
        - **Octave Feature Representation** that explicitly factorizes the feature map tensors into groups corresponding to low and high frequencies.
        - The scale-space theory provides a principled way of creating scale-spaces of spatial resolutions and defines an octave as a division of the spatial dimensions by a power of 2 (we only explore 2¹ in this work).

            [Scale space](https://en.wikipedia.org/wiki/Scale_space)

- Octave Convolution

    After the reducion of the spatial redundancy there's still a problem:

    - PROBLEM: A vanilla convolution cannot directly operate on this new representation due, to differences in spatial resolution in the input features.
    - SOLUTION
        - The goal is to effectively process the low and high frequency in their corresponding frequency tensor but also enable efficient inter-frequency communication.
        - X, Y are factorized input and output tensors. The high and low frequency feature map of the output: Y = {Y^h, Y^L} will be

            $$Y^H = Y^{H -> H} + Y^{L->H}  \thinspace \thinspace and \thinspace \thinspace   Y^L = Y^{L -> L} + Y^{H->L}$$

            Where → denotes a convolutional update

        - To compute these terms, the kernel is divided into two components: W^h and W^L

            ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%201.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%201.png)

        - Specifically for high-frequency feature map, we compute it at location (p, q)

            ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%202.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%202.png)

            ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%203.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%203.png)

            To assure that X^h is an integer it's possible to perform a strided convolution or a **average pooling**

- Implementation Details
    - Now output Y = {Y^h, Y^L} of the Octave Convolution using average pooling

        ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%204.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%204.png)

        ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%205.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%205.png)

    - The paths with **green color, denot change the spatial dimensions going from input to output**. However, the paths with r**ed color either increase (Low-to-high) or decrease (High-to-low) the spatial dimensions going from input to output**.
    - When going from **high frequency input to low frequency output** (HtoL path), a **2x2 pooling operation** is done to get the **downscaled** **input** **for** **convolution**.
    - When going from **Low-Frequency input to high-frequency output** (LtoH path), a vanilla convolution is topped with **bilinear interpolation to upsample the low-resolution conv output.**
    - At the heart of Octave convolution lies the concept of α (ratio of the total channels which are used by low-frequency convolutions). For the **first convolution layer, there is no low-frequency input channel, so α_{in} = 0**. Similarly, **for the last convolution layer, there is no low-frequency output channel, α_{out} = 0**. For all the other layers, the authors assumed α_{in} = α_{out} = 0.5.
    - Group and Depth-wise convolutions
        - The **Octave Convolution can also be adopted to other popular variants of vanilla convolution such as group or depth-wise.**
        - For the group convolution case, simply set all four convolution operations that appear inside the design of the OctConv to group convolutions. Similarly, for the depth-wise convolution case, the convolution operations are depth-wise and therefore the information exchange paths are eliminated, leaving only two depth-wise convolution operations.
    - Efficiency analysis.

        ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%206.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%206.png)

    - Integrating OctConv into backbone networks

        OctConv is backwards compatible with vanilla convolution and can be inserted to regular convolutional networks without special adjustment.

    - Comparison to Multi-grid Convolution
        - The multi-grid conv is a bi-directional and cross-scale convolution operator. Though being conceptually similar the OctConv is different from MG-Conv in both the core motivation and design.
        - In terms of design, MG-Conv adopts **max-pooling for down-sampling**.
        - OctConv aims for reducing spatial redundancy and is a naive extension of convolution operator. It uses **average pooling** to **distill low-frequency features** without extra memory cost and its upsampling operation follows the convolution

### Experimental Evaluation

- Experimental Setups
    - Replaced the regular convolutions with OctConv (except the first convolutional layer before the max pooling).
    - The resulting networks only have one global hyper-parameter α, which denotes the ratio of low frequency part.
    - All networks are trained with naı̈ve softmax cross entropy loss except that the **MobileNetV2 also adopts the label smoothing**
- Results

    ![Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%207.png](Drop%20an%20Octave%20Reducing%20Spatial%20Redundancy%20in%20Conv/Untitled%207.png)

### Conclusion

- The authors propose a novel Octave Convolution operation to store and process low- and high-frequency features separately to improve the model efficiency
- Octave Convolution is sufficiently generic to replace the regular convolution operation in-place, and can be used in most 2D and 3D CNNs without model architecture adjustment
- Beyond saving a substantial amount of computation and memory, Octave Convolution can also improve the recognition performance
