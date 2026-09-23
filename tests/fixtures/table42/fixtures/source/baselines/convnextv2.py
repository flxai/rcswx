# The ConvNeXt_v2 architecture, represented in einspace
# the depthwise 7×7 convolution is replaced with a standard 7×7 convolution,
# The GELU activation function is replaced with ReLU,
# Global Response Normalization is omitted
# The LayerNorm is replaced with a standard norm.


convnextv2_stem = """
    sequential[
        routing[im2col4k4s0p, computation[linear96], col2im],
        computation[norm]
    ]"""

convnextv2_block = lambda c, e: f"""
    branching(2)[
        clone(2),
        routing[im2col7k1s3p, 
                sequential[
                    sequential[computation[{c}],
                                computation[norm]
                                ],
                    sequential[
                        computation[{e}],
                        sequential[
                            computation[relu],
                            computation[{c}]
                        ]
                    ]
                ]
                col2im],
        computation[identity],
        add(2)
    ]"""

convnextv2_downsample = lambda c: f"""
    sequential[
        computation[norm],
        routing[im2col2k2s0p, computation[{c}], col2im]
    ]"""

convnextv2_tiny = f"""
    sequential[
        sequential[
            sequential[
                {convnextv2_stem},
                {convnextv2_block("linear96", "linear384")}
            ],
            sequential[
                sequential[
                    {convnextv2_block("linear96", "linear384")},
                    {convnextv2_block("linear96", "linear384")}
                ],
                sequential[
                    {convnextv2_downsample("linear192")},
                    sequential[
                        {convnextv2_block("linear192", "linear768")},
                        sequential[
                            {convnextv2_block("linear192", "linear768")},
                            {convnextv2_block("linear192", "linear768")}
                        ]
                    ]
                ]
            ]
        ],
        sequential[
            sequential[
                {convnextv2_downsample("linear384")},
                sequential[
                    sequential[
                        {convnextv2_block("linear384", "linear1536")},
                        {convnextv2_block("linear384", "linear1536")}
                    ],
                    sequential[
                        sequential[
                            {convnextv2_block("linear384", "linear1536")},
                            {convnextv2_block("linear384", "linear1536")}
                        ],
                        sequential[
                            {convnextv2_block("linear384", "linear1536")},
                            sequential[
                                {convnextv2_block("linear384", "linear1536")},
                                sequential[
                                    {convnextv2_block("linear384", "linear1536")},
                                    computation[identity]
                                ]
                            ]
                        ]
                    ]
                ]
            ],
            sequential[
                {convnextv2_downsample("linear768")},
                sequential[
                    {convnextv2_block("linear768", "linear3072")},
                    sequential[
                        {convnextv2_block("linear768", "linear3072")},
                        {convnextv2_block("linear768", "linear3072")}
                    ]
                ]
            ]
            
        ]
    ]"""