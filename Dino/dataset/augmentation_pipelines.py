from imgaug import augmenters as iaa


def get_augmentation_pipeline(augmentation_severity=1):
    """
    Defining the augmentation pipeline for SemiMTR pre-training and fine-tuning.
    :param augmentation_severity:
    :return: augmentation_pipeline
    """
    if augmentation_severity == 1:
        augmentations = iaa.Sequential([
            iaa.Invert(0.5),
            iaa.OneOf([
                iaa.ChannelShuffle(0.35),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.KMeansColorQuantization(),
                iaa.HistogramEqualization(),
                iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                iaa.ChangeColorTemperature((1100, 10000))
            ]),
            iaa.OneOf([
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                iaa.OneOf([
                    iaa.GaussianBlur((0.5, 1.5)),
                    iaa.AverageBlur(k=(2, 6)),
                    iaa.MedianBlur(k=(3, 7)),
                    iaa.MotionBlur(k=5)
                ])
            ]),
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.ImpulseNoise(0.1),
                iaa.MultiplyElementwise((0.5, 1.5))
            ])
        ])
    elif augmentation_severity == 2:
        optional_augmentations_list = [
            iaa.LinearContrast((0.5, 1.0)),
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Crop(percent=((0, 0.4), (0, 0), (0, 0.4), (0, 0.0)), keep_size=True),
            iaa.Crop(percent=((0, 0.0), (0, 0.02), (0, 0), (0, 0.02)), keep_size=True),
            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
            # iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'), # In SeqCLR but replaced with a faster aug
            iaa.ElasticTransformation(alpha=(0, 0.8), sigma=0.25),
            iaa.PerspectiveTransform(scale=(0.01, 0.02)),
        ]
        augmentations = iaa.SomeOf((1, None), optional_augmentations_list, random_order=True)
    elif augmentation_severity == 3:
        augmentations = iaa.Sequential([
            iaa.Invert(0.1),
            iaa.OneOf([
                iaa.ChannelShuffle(0.35),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.KMeansColorQuantization(),
                iaa.HistogramEqualization(),
                iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                iaa.ChangeColorTemperature((1100, 10000))
            ]),
            iaa.OneOf([
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                iaa.OneOf([
                    iaa.GaussianBlur((0.5, 1.5)),
                    iaa.AverageBlur(k=(2, 6)),
                    iaa.MedianBlur(k=(3, 7)),
                    iaa.MotionBlur(k=5)
                ])
            ]),
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.ImpulseNoise(0.1),
                iaa.MultiplyElementwise((0.5, 1.5))
            ])
        ])
    elif augmentation_severity == 4:
        augmentations_1 = iaa.Sometimes(0.3, iaa.Invert(0.1))
        augmentations_2 = iaa.Sometimes(0.6, iaa.OneOf([
            iaa.ChannelShuffle(0.35),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.KMeansColorQuantization(),
            iaa.HistogramEqualization(),
            iaa.CLAHE(),
            iaa.Dropout(p=(0, 0.1), per_channel=0.5),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.LinearContrast((0.5, 1.0)),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.ChangeColorTemperature((1100, 10000))
        ]))
        augmentations_3 = iaa.Sometimes(0.6, iaa.OneOf([
            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
            iaa.OneOf([
                iaa.GaussianBlur((0.5, 1.5)),
                iaa.AverageBlur(k=(2, 6)),
                iaa.MedianBlur(k=(3, 7)),
                iaa.MotionBlur(k=5)
            ])
        ]))
        augmentations_4 = iaa.Sometimes(0.6, iaa.OneOf([
            iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
            iaa.ImpulseNoise(0.05),
            iaa.MultiplyElementwise((0.5, 1.5)),
            iaa.CoarseDropout(0.02, size_percent=0.5)
        ]))
        augmentations = iaa.Sometimes(0.2,
                                      iaa.Identity(),
                                      iaa.Sequential([
                                          augmentations_1,
                                          augmentations_2,
                                          augmentations_3,
                                          augmentations_4
                                      ])
                                      )
    elif augmentation_severity == 5:
        probability = [0.7, 0.2]
        arithmetic = iaa.OneOf([
            iaa.AddElementwise((-40, 40)),  # 0.020992517471313477
            iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),  # 0.03757429122924805
            iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)),  # 0.028045654296875
            iaa.AdditivePoissonNoise(lam=(0, 40)),  # 0.02863311767578125
            iaa.Multiply((0.5, 1.5), per_channel=0.5),  # 0.011006593704223633
            iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),  # 0.022083520889282227
            iaa.Dropout(p=(0, 0.1), per_channel=0.5),  # 0.013317108154296875
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),  # 0.02934122085571289
            iaa.Dropout2d(p=0.5),  # 0.00327301025390625
            iaa.ImpulseNoise(0.1),  # 0.019927501678466797
            iaa.SaltAndPepper(0.1),  # 0.02179861068725586
            iaa.Salt(0.1),  # 0.0442655086517334
            iaa.Pepper(0.1),  # 0.023215055465698242
            iaa.Invert(0.15),  # 0.0016281604766845703
            iaa.Solarize(0.5, threshold=(32, 128)),  # 0.0045392513275146484
            iaa.JpegCompression(compression=(70, 99)),  # 0.06881833076477051
            iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),  # 0.01827859878540039
            iaa.EdgeDetect(alpha=(0.0, 1.0)),  # 0.01613450050354004
            iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)),  # 0.06784319877624512
            iaa.pillike.FilterEdgeEnhanceMore(),  # 0.008636713027954102
            iaa.pillike.FilterContour(),  # 0.008586645126342773
        ])
        color = iaa.Sometimes(probability[0],
                              iaa.OneOf([
                                  iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                                     children=iaa.WithChannels(0, iaa.Add((0, 50)))),  # 0.007066011428833008
                                  iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),  # 0.010883331298828125
                                  iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),  # 0.00563359260559082
                                  iaa.AddToHueAndSaturation((-50, 50), per_channel=True),  # 0.01298975944519043
                                  iaa.Sequential([
                                      iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                      iaa.WithChannels(0, iaa.Add((50, 100))),
                                      iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
                                  ]),  # 0.002664804458618164
                                  iaa.Grayscale(alpha=(0.0, 1.0)),  # 0.0020821094512939453
                                  iaa.KMeansColorQuantization(),  # 0.13840675354003906
                                  iaa.UniformColorQuantization(),  # 0.004789829254150391
                                  iaa.ChangeColorTemperature((1100, 10000))  # 0.0026831626892089844
                              ]))
        Blur = iaa.Sometimes(probability[0], iaa.OneOf([
            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),  # 0.0017418861389160156
            iaa.OneOf([
                iaa.GaussianBlur((0.5, 1.5)),  # 0.0033369064331054688
                iaa.AverageBlur(k=(2, 6)),  # 0.0012645721435546875
                iaa.MedianBlur(k=(3, 7)),  # 0.0016665458679199219
                iaa.MotionBlur(k=5),  # 0.0035009384155273438
                # iaa.MeanShiftBlur(),  # 0.5308325290679932
                iaa.BilateralBlur(
                    d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),  # 0.003159046173095703
            ])
        ]))
        contrast = iaa.Sometimes(probability[0], iaa.OneOf([
            iaa.GammaContrast((0.5, 2.0)),  # 0.0015556812286376953
            iaa.LinearContrast((0.5, 1.0)),  # 0.001466512680053711
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),  # 0.001722097396850586
            iaa.LogContrast(gain=(0.6, 1.4)),  # 0.0016601085662841797
            iaa.HistogramEqualization(),  # 0.001706838607788086
            iaa.AllChannelsHistogramEqualization(),  # 0.0014772415161132812
            iaa.CLAHE(),  # 0.009737253189086914
            iaa.AllChannelsCLAHE(),  # 0.012245655059814453
        ]))
        weather = iaa.Sometimes(probability[0], iaa.OneOf([
            # iaa.imgcorruptlike.Fog(severity=2),
            # iaa.imgcorruptlike.Frost(severity=2),
            # iaa.imgcorruptlike.Snow(severity=2),
            # iaa.imgcorruptlike.Spatter(severity=2),
            # iaa.imgcorruptlike.Pixelate(severity=2),
            iaa.Fog(),  # 0.009767293930053711
            iaa.Clouds(),  # 0.020981788635253906
            iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),  # 0.024015426635742188
            iaa.Rain(speed=(0.1, 0.3)),  # 0.02486562728881836
        ]))
        augmentations = iaa.Sometimes(probability[1],
                                      iaa.Identity(),
                                      iaa.Sequential([
                                          arithmetic,
                                          color,
                                          Blur,
                                          contrast,
                                          weather,
                                      ])
                                      )
    elif augmentation_severity == 6:
        augmentations = iaa.OneOf([
            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add((0, 50)))),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((50, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
            ]),
            iaa.UniformColorQuantization(),
            iaa.ChannelShuffle(0.35),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.KMeansColorQuantization(),
            iaa.HistogramEqualization(),
            iaa.Dropout(p=(0, 0.2), per_channel=0.5),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.ChangeColorTemperature((1100, 10000)),
            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
            iaa.CLAHE(),
            iaa.LinearContrast((0.5, 1.0)),
        ])
    else:
        raise NotImplementedError(f'augmentation_severity={augmentation_severity} is not supported')

    return augmentations
