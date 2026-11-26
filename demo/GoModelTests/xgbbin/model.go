package xgbbin

import "math"

const (
	baseScore = -0.5753641449035618
)

// PredictScore returns the raw logit for feature vector f.
func PredictScore(f []float64) float64 {
	s := baseScore
	s += tree0(f)
	s += tree1(f)
	s += tree2(f)
	s += tree3(f)
	s += tree4(f)
	s += tree5(f)
	s += tree6(f)
	s += tree7(f)
	s += tree8(f)
	s += tree9(f)
	s += tree10(f)
	s += tree11(f)
	s += tree12(f)
	s += tree13(f)
	s += tree14(f)
	s += tree15(f)
	s += tree16(f)
	s += tree17(f)
	s += tree18(f)
	s += tree19(f)
	s += tree20(f)
	s += tree21(f)
	s += tree22(f)
	s += tree23(f)
	s += tree24(f)
	s += tree25(f)
	s += tree26(f)
	s += tree27(f)
	s += tree28(f)
	s += tree29(f)
	s += tree30(f)
	s += tree31(f)
	s += tree32(f)
	s += tree33(f)
	s += tree34(f)
	s += tree35(f)
	s += tree36(f)
	s += tree37(f)
	s += tree38(f)
	s += tree39(f)
	return s
}

// PredictProbability returns sigmoid(PredictScore(f)).
func PredictProbability(f []float64) float64 {
	x := PredictScore(f)
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	return math.Exp(x) / (1.0 + math.Exp(x))
}

// Predict returns 0 or 1 (threshold 0.5).
func Predict(f []float64) int {
	if PredictScore(f) >= 0 {
		return 1
	}
	return 0
}

func tree0(f []float64) float64 {
	if float32(f[5]) < -0.136334032 {
		if float32(f[7]) < -3.84925079 {
			if float32(f[8]) < -0.356379688 {
				return -0.10129431999999999
			}
			return 0.19370459000000001
		}
		if float32(f[7]) < 0.2178372 {
			return -0.13595428000000001
		}
		return 0.0042205923000000001
	}
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.06774592 {
			return 0.035871717999999997
		}
		return 0.20438294000000001
	}
	if float32(f[3]) < 2.37301254 {
		return 0.23811521999999999
	}
	return 0.018371091999999999
}

func tree1(f []float64) float64 {
	if float32(f[5]) < -0.136334032 {
		if float32(f[7]) < -3.84925079 {
			if float32(f[8]) < -0.356379688 {
				return -0.096720780000000006
			}
			return 0.17421619999999999
		}
		if float32(f[7]) < 0.2178372 {
			return -0.12818626999999999
		}
		return 0.003943808
	}
	if float32(f[5]) < 1.20587265 {
		if float32(f[4]) < -1.3991549 {
			return -0.012487724
		}
		return 0.11951691
	}
	if float32(f[3]) < 2.37301254 {
		return 0.20607028999999999
	}
	return 0.01718035
}

func tree2(f []float64) float64 {
	if float32(f[5]) < -0.136334032 {
		if float32(f[7]) < -3.56349397 {
			if float32(f[8]) < -0.356379688 {
				return -0.10209905
			}
			return 0.168708
		}
		if float32(f[7]) < 0.2178372 {
			return -0.12433565000000001
		}
		return 0.0036852774
	}
	if float32(f[5]) < 0.843082607 {
		if float32(f[3]) < 2.05752087 {
			return 0.010317251
		}
		return 0.16614892000000001
	}
	if float32(f[8]) < 2.13314939 {
		return 0.17143217999999999
	}
	return -0.051313656999999999
}

func tree3(f []float64) float64 {
	if float32(f[5]) < -0.339402914 {
		if float32(f[7]) < -3.84925079 {
			if float32(f[8]) < -0.356379688 {
				return -0.088426149999999995
			}
			return 0.13594935999999999
		}
		if float32(f[5]) < -0.572647333 {
			return -0.12446735
		}
		return -0.035570886000000003
	}
	if float32(f[5]) < 0.532531202 {
		if float32(f[3]) < 2.05752087 {
			return -0.016033354999999999
		}
		return 0.13554548
	}
	if float32(f[8]) < 2.03907418 {
		return 0.13730107
	}
	return -0.034093252999999997
}

func tree4(f []float64) float64 {
	if float32(f[5]) < -0.339402914 {
		if float32(f[7]) < -3.84925079 {
			if float32(f[8]) < -0.356379688 {
				return -0.08498762
			}
			return 0.12705730000000001
		}
		if float32(f[5]) < -0.572647333 {
			return -0.119353615
		}
		return -0.03327277
	}
	if float32(f[5]) < 0.843082607 {
		if float32(f[6]) < 0.4662067 {
			return -0.017041405999999999
		}
		return 0.1068692
	}
	if float32(f[8]) < 2.13314939 {
		return 0.14098874
	}
	return -0.047025079999999997
}

func tree5(f []float64) float64 {
	if float32(f[5]) < -0.339402914 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[6]) < -2.39232111 {
				return -0.021867909000000001
			}
			return -0.1206271
		}
		if float32(f[7]) < -2.30440378 {
			return 0.12326735
		}
		return -0.105775654
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[6]) < 0.286042154 {
			return 0.0031659330000000001
		}
		return 0.10890682
	}
	return 0.14907253000000001
}

func tree6(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[3]) < 0.908073723 {
				return -0.11583543
			}
			return 0.011450273
		}
		if float32(f[7]) < -3.34253025 {
			return 0.14151905000000001
		}
		return -0.036117967000000001
	}
	if float32(f[1]) < -0.689727783 {
		if float32(f[2]) < -1.10269809 {
			return 0.095312789999999994
		}
		return -0.048163153
	}
	if float32(f[8]) < 2.13314939 {
		return 0.106258646
	}
	return -0.002480524
}

func tree7(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[7]) < -4.02032995 {
			if float32(f[8]) < -0.356379688 {
				return -0.064404569999999994
			}
			return 0.094486915000000005
		}
		if float32(f[6]) < -2.39232111 {
			return -0.015487199
		}
		return -0.11229264
	}
	if float32(f[6]) < 0.646803975 {
		if float32(f[9]) < 1.06287766 {
			return -0.01420167
		}
		return 0.1440179
	}
	if float32(f[9]) < 1.11356258 {
		return 0.1470186
	}
	return -0.060245163999999997
}

func tree8(f []float64) float64 {
	if float32(f[5]) < -0.136334032 {
		if float32(f[7]) < -3.34253025 {
			if float32(f[8]) < -0.356379688 {
				return -0.086372845000000004
			}
			return 0.12971567000000001
		}
		if float32(f[7]) < 0.2178372 {
			return -0.10394245000000001
		}
		return 0.0095999799999999993
	}
	if float32(f[6]) < 2.17065048 {
		if float32(f[6]) < 0.760009587 {
			return 0.040977906000000001
		}
		return 0.13587646
	}
	return -0.10069292000000001
}

func tree9(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[3]) < 0.908073723 {
				return -0.10725164399999999
			}
			return 0.0082255739999999994
		}
		if float32(f[6]) < 1.17189097 {
			return -0.052301206000000003
		}
		return 0.093349165999999997
	}
	if float32(f[5]) < 1.20587265 {
		if float32(f[0]) < 0.818777621 {
			return -0.0031500425999999998
		}
		return 0.11802821600000001
	}
	if float32(f[3]) < 2.37301254 {
		return 0.13069210000000001
	}
	return -0.0070256772000000002
}

func tree10(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[7]) < -4.02032995 {
			if float32(f[4]) < -2.38948131 {
				return -0.035881132000000003
			}
			return 0.074657710000000002
		}
		if float32(f[7]) < 0.0505226403 {
			return -0.10403839500000001
		}
		return -0.011424703
	}
	if float32(f[5]) < 1.20587265 {
		if float32(f[6]) < 0.646803975 {
			return -0.028432005999999999
		}
		return 0.087212645000000005
	}
	if float32(f[3]) < 2.37301254 {
		return 0.12347207
	}
	return -0.006588727
}

func tree11(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[7]) < -4.02032995 {
			if float32(f[6]) < 2.52663636 {
				return -0.035604524999999998
			}
			return 0.072771749999999996
		}
		if float32(f[6]) < -2.05104065 {
			return -0.015655885000000001
		}
		return -0.10127985
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[6]) < 0.646803975 {
			return -0.013472731
		}
		return 0.079698989999999997
	}
	if float32(f[1]) < 0.813205183 {
		return 0.13152829999999999
	}
	return 0.035880769999999999
}

func tree12(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[7]) < 0.2178372 {
				return -0.10450922999999999
			}
			return -0.013516792999999999
		}
		if float32(f[6]) < 1.17189097 {
			return -0.045005415
		}
		return 0.084264909999999998
	}
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.06774592 {
			return -0.0056013069999999998
		}
		return 0.11526346
	}
	if float32(f[3]) < 2.37301254 {
		return 0.11583381
	}
	return -0.0085104610000000004
}

func tree13(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[7]) < -4.02032995 {
			return 0.023733890000000001
		}
		if float32(f[0]) < 1.18891335 {
			return -0.097242609999999993
		}
		return -0.011560567000000001
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[6]) < 0.646803975 {
			return -0.014148610000000001
		}
		return 0.072266944
	}
	if float32(f[1]) < 0.813205183 {
		return 0.12246213
	}
	return 0.032018105999999998
}

func tree14(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[7]) < 0.2178372 {
				return -0.100954376
			}
			return -0.012642296000000001
		}
		if float32(f[6]) < 1.17189097 {
			return -0.04075906
		}
		return 0.077233999999999997
	}
	if float32(f[0]) < 0.818777621 {
		if float32(f[5]) < 0.647607148 {
			return -0.036277186000000003
		}
		return 0.059944230000000001
	}
	return 0.11077531
}

func tree15(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[6]) < 2.20873284 {
			if float32(f[6]) < -1.89064252 {
				return -0.012148383
			}
			return -0.10064866
		}
		if float32(f[8]) < 0.364357382 {
			return -0.067934049999999996
		}
		return 0.073961936000000006
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[2]) < 0.152406931 {
			return -0.021305266999999999
		}
		return 0.055079309999999999
	}
	if float32(f[1]) < 0.813205183 {
		return 0.11518513399999999
	}
	return 0.028537197
}

func tree16(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[3]) < 0.908073723 {
				return -0.093271370000000006
			}
			return 0.0049116029999999996
		}
		if float32(f[6]) < 1.17189097 {
			return -0.036865145000000002
		}
		return 0.070545129999999998
	}
	if float32(f[0]) < 0.818777621 {
		if float32(f[0]) < -0.1758544 {
			return 0.055177654999999999
		}
		return -0.033775367000000001
	}
	if float32(f[7]) < 0.815552294 {
		return 0.11118450000000001
	}
	return 0.032381859999999998
}

func tree17(f []float64) float64 {
	if float32(f[7]) < 0.274200141 {
		if float32(f[8]) < 0.667639375 {
			if float32(f[1]) < 0.584412992 {
				return -0.093085000000000001
			}
			return 0.023444025
		}
		if float32(f[2]) < 0.099002853 {
			return -0.014272757
		}
		return 0.09891664
	}
	if float32(f[4]) < 1.03198743 {
		if float32(f[8]) < -0.750269353 {
			return -0.0023037029999999998
		}
		return 0.11402517
	}
	if float32(f[6]) < 0.187009752 {
		return -0.082856579999999999
	}
	return 0.071376376000000005
}

func tree18(f []float64) float64 {
	if float32(f[5]) < 0.843082607 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.16915679 {
				return -0.10015656000000001
			}
			return 0.0090364910000000007
		}
		if float32(f[2]) < 0.0516255088 {
			return 0.0015801176
		}
		return 0.12391515
	}
	if float32(f[8]) < 2.13314939 {
		if float32(f[6]) < -1.78869879 {
			return -0.0074519479999999999
		}
		return 0.104431145
	}
	return -0.047788110000000002
}

func tree19(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[6]) < 2.20873284 {
			if float32(f[6]) < -1.81492484 {
				return -0.0090482370000000006
			}
			return -0.095601610000000004
		}
		if float32(f[8]) < 0.364357382 {
			return -0.061664056000000002
		}
		return 0.065941330000000006
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[6]) < 0.225191504 {
			return -0.021838969
		}
		return 0.051762242
	}
	if float32(f[1]) < 0.742682397 {
		return 0.10491808
	}
	return 0.02876244
}

func tree20(f []float64) float64 {
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.4662067 {
				return -0.082081370000000001
			}
			return 0.019809494
		}
		if float32(f[2]) < 0.0516255088 {
			return -0.0020765783
		}
		return 0.12190222000000001
	}
	if float32(f[3]) < 2.37301254 {
		if float32(f[6]) < -1.42282522 {
			return 0.019131582000000001
		}
		return 0.10853892599999999
	}
	return -0.022662615000000001
}

func tree21(f []float64) float64 {
	if float32(f[7]) < 0.274200141 {
		if float32(f[8]) < 0.667639375 {
			if float32(f[1]) < 0.584412992 {
				return -0.086811029999999997
			}
			return 0.025448749
		}
		if float32(f[1]) < 0.742682397 {
			return 0.053550359999999998
		}
		return -0.068313739999999998
	}
	if float32(f[4]) < 1.03198743 {
		if float32(f[9]) < -1.17356002 {
			return -0.014743141
		}
		return 0.10096742
	}
	if float32(f[3]) < 1.56040657 {
		return 0.050189632999999997
	}
	return -0.09028854
}

func tree22(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[7]) < -4.02032995 {
			return 0.026644055
		}
		if float32(f[7]) < -0.314106822 {
			return -0.083582975000000004
		}
		return -0.0080539575000000002
	}
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.06774592 {
			return -0.015290021000000001
		}
		return 0.088768429999999995
	}
	if float32(f[3]) < 2.37301254 {
		return 0.093790970000000001
	}
	return -0.020197676000000001
}

func tree23(f []float64) float64 {
	if float32(f[5]) < 0.532531202 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.16915679 {
				return -0.095095865000000002
			}
			return 0.0028136414000000001
		}
		if float32(f[2]) < 0.0516255088 {
			return -0.0055836528
		}
		return 0.101720825
	}
	if float32(f[1]) < 0.813205183 {
		if float32(f[6]) < 0.286042154 {
			return 0.031930069999999998
		}
		return 0.10571911000000001
	}
	if float32(f[1]) < 1.23894632 {
		return -0.083930190000000002
	}
	return 0.062939010000000004
}

func tree24(f []float64) float64 {
	if float32(f[5]) < -0.604840755 {
		if float32(f[6]) < 2.20873284 {
			if float32(f[7]) < -0.333554149 {
				return -0.090066834999999998
			}
			return -0.0071498380000000004
		}
		if float32(f[8]) < 0.364357382 {
			return -0.056840429999999997
		}
		return 0.062293567000000001
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[7]) < -2.07632375 {
			return 0.073596170000000002
		}
		return -0.011056313
	}
	if float32(f[1]) < 0.557691038 {
		return 0.096248769999999997
	}
	return 0.025657606999999999
}

func tree25(f []float64) float64 {
	if float32(f[5]) < 0.843082607 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.16915679 {
				return -0.086391480000000007
			}
			return 0.0093520129999999993
		}
		if float32(f[2]) < 0.0516255088 {
			return -0.0079049229999999995
		}
		return 0.10107519500000001
	}
	if float32(f[8]) < 0.397235602 {
		return 0.1035509
	}
	if float32(f[1]) < -0.998060644 {
		return -0.078326783999999997
	}
	return 0.039228495000000002
}

func tree26(f []float64) float64 {
	if float32(f[7]) < 0.274200141 {
		if float32(f[9]) < 0.612526476 {
			if float32(f[6]) < 0.881998897 {
				return -0.040099113999999998
			}
			return 0.112652265
		}
		if float32(f[0]) < 0.975816727 {
			return -0.10471133000000001
		}
		return 0.014322914000000001
	}
	if float32(f[4]) < 1.03198743 {
		if float32(f[9]) < -1.14881313 {
			return -0.011756957
		}
		return 0.090293819999999997
	}
	if float32(f[3]) < 1.56040657 {
		return 0.045159052999999998
	}
	return -0.085555450000000005
}

func tree27(f []float64) float64 {
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.4662067 {
				return -0.069562239999999997
			}
			return 0.018580083000000001
		}
		if float32(f[8]) < 2.51316977 {
			return 0.099430710000000005
		}
		return 0.00012816720000000001
	}
	if float32(f[3]) < 2.37301254 {
		if float32(f[1]) < -0.650023937 {
			return 0.019344633
		}
		return 0.096013210000000002
	}
	return -0.021975215999999999
}

func tree28(f []float64) float64 {
	if float32(f[5]) < -0.0897523835 {
		if float32(f[7]) < -2.30440378 {
			if float32(f[8]) < 0.0459972806 {
				return -0.065177180000000001
			}
			return 0.07443002
		}
		if float32(f[7]) < 0.2178372 {
			return -0.10029258000000001
		}
		return 0.034558541999999998
	}
	if float32(f[6]) < 2.12865591 {
		if float32(f[9]) < 1.06287766 {
			return 0.011812361
		}
		return 0.10087635
	}
	return -0.068933315999999994
}

func tree29(f []float64) float64 {
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.16915679 {
				return -0.073396644999999996
			}
			return 0.012796715
		}
		if float32(f[8]) < 2.51316977 {
			return 0.093289010000000006
		}
		return -0.0016925877
	}
	if float32(f[3]) < 2.37301254 {
		if float32(f[6]) < -1.25840855 {
			return 0.017738719999999999
		}
		return 0.093458490000000005
	}
	return -0.021362202
}

func tree30(f []float64) float64 {
	if float32(f[5]) < 0.532531202 {
		if float32(f[7]) < -2.30440378 {
			if float32(f[9]) < 0.612526476 {
				return 0.074185039999999994
			}
			return -0.059352387
		}
		if float32(f[7]) < 0.230475456 {
			return -0.074359775000000003
		}
		return 0.028040651
	}
	if float32(f[1]) < 0.813205183 {
		if float32(f[6]) < 0.286042154 {
			return 0.027123940999999999
		}
		return 0.095302570000000003
	}
	if float32(f[1]) < 1.23894632 {
		return -0.081376089999999998
	}
	return 0.054026461999999997
}

func tree31(f []float64) float64 {
	if float32(f[7]) < 1.50645638 {
		if float32(f[4]) < -3.90911245 {
			return 0.085505540000000005
		}
		if float32(f[6]) < 0.16915679 {
			return -0.044358163999999999
		}
		return 0.011614217
	}
	if float32(f[6]) < -0.470468402 {
		return 0.086953150000000007
	}
	return 0.01899085
}

func tree32(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[8]) < 0.392289519 {
			if float32(f[7]) < 0.2178372 {
				return -0.082156690000000004
			}
			return 0.0053166276999999998
		}
		if float32(f[7]) < -2.5433321 {
			return 0.062764849999999997
		}
		return -0.032112162999999999
	}
	if float32(f[0]) < 0.818777621 {
		if float32(f[0]) < -0.1758544 {
			return 0.038402079999999998
		}
		return -0.036081200000000001
	}
	if float32(f[6]) < 0.0523812063 {
		return 0.030627598999999998
	}
	return 0.096682599999999994
}

func tree33(f []float64) float64 {
	if float32(f[5]) < -1.15118361 {
		if float32(f[6]) < 2.20873284 {
			return -0.079280630000000005
		}
		return 0.0027189672000000002
	}
	if float32(f[7]) < 1.50645638 {
		if float32(f[7]) < -2.07632375 {
			return 0.053844202000000001
		}
		return -0.010654825
	}
	if float32(f[6]) < -0.470468402 {
		return 0.083862229999999996
	}
	return 0.016755803
}

func tree34(f []float64) float64 {
	if float32(f[5]) < 0.843082607 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.16915679 {
				return -0.072755120000000006
			}
			return 0.0074173809999999998
		}
		if float32(f[8]) < 2.51316977 {
			return 0.085235909999999998
		}
		return -0.007453777
	}
	if float32(f[8]) < 0.397235602 {
		return 0.0956177
	}
	if float32(f[9]) < -0.259361565 {
		return -0.040125220000000003
	}
	return 0.049572404
}

func tree35(f []float64) float64 {
	if float32(f[5]) < -0.00745649543 {
		if float32(f[7]) < -3.34253025 {
			if float32(f[8]) < 0.523042142 {
				return -0.014938978
			}
			return 0.070443994999999995
		}
		if float32(f[7]) < 0.2178372 {
			return -0.064741129999999994
		}
		return 0.026341110000000001
	}
	if float32(f[0]) < 0.818777621 {
		if float32(f[4]) < -3.90911245 {
			return 0.096045400000000003
		}
		return -0.0068406513999999998
	}
	if float32(f[6]) < 0.0523812063 {
		return 0.028080007000000001
	}
	return 0.091861083999999996
}

func tree36(f []float64) float64 {
	if float32(f[5]) < -1.00868821 {
		if float32(f[6]) < 2.20873284 {
			return -0.078209824999999997
		}
		return 0.011338862999999999
	}
	if float32(f[9]) < -2.19332671 {
		if float32(f[6]) < -1.28273475 {
			return -0.071587170000000006
		}
		return -0.0075767203
	}
	if float32(f[6]) < -2.23088884 {
		return 0.098285220000000006
	}
	return 0.011845277
}

func tree37(f []float64) float64 {
	if float32(f[5]) < 1.20587265 {
		if float32(f[3]) < 2.14226198 {
			if float32(f[6]) < 0.881998897 {
				return -0.049388077000000002
			}
			return 0.028412303
		}
		if float32(f[2]) < 0.584888756 {
			return -0.0081827400000000008
		}
		return 0.085486129999999994
	}
	if float32(f[3]) < 2.37301254 {
		if float32(f[6]) < -1.23875487 {
			return 0.017306809999999999
		}
		return 0.086142590000000005
	}
	return -0.026663307000000001
}

func tree38(f []float64) float64 {
	if float32(f[7]) < 0.274200141 {
		if float32(f[9]) < 0.612526476 {
			if float32(f[6]) < 0.881998897 {
				return -0.028861293999999999
			}
			return 0.1005543
		}
		if float32(f[0]) < 0.975816727 {
			return -0.10052182
		}
		return 0.013785866000000001
	}
	if float32(f[4]) < 1.03198743 {
		if float32(f[0]) < -0.1758544 {
			return 0.10590920600000001
		}
		return 0.022831079000000001
	}
	if float32(f[3]) < 1.56040657 {
		return 0.036657943999999998
	}
	return -0.083481810000000004
}

func tree39(f []float64) float64 {
	if float32(f[4]) < -3.90911245 {
		return 0.078532119999999997
	}
	if float32(f[7]) < 0.274200141 {
		if float32(f[9]) < -0.707715094 {
			return 0.026205009000000001
		}
		return -0.054673220000000002
	}
	if float32(f[4]) < 1.03198743 {
		return 0.061884790000000002
	}
	return -0.027074924
}
