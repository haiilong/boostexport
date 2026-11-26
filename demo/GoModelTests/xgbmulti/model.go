package xgbmulti

import "math"

// PredictScores returns raw per-class scores.
func PredictScores(f []float64) []float64 {
	s0 := -0.030223726999999999
	s1 := 0.020038008999999999
	s2 := 0.010185718999999999
	s0 += tree0(f)
	s0 += tree3(f)
	s0 += tree6(f)
	s0 += tree9(f)
	s0 += tree12(f)
	s0 += tree15(f)
	s0 += tree18(f)
	s0 += tree21(f)
	s0 += tree24(f)
	s0 += tree27(f)
	s0 += tree30(f)
	s0 += tree33(f)
	s0 += tree36(f)
	s0 += tree39(f)
	s0 += tree42(f)
	s0 += tree45(f)
	s0 += tree48(f)
	s0 += tree51(f)
	s0 += tree54(f)
	s0 += tree57(f)
	s0 += tree60(f)
	s0 += tree63(f)
	s0 += tree66(f)
	s0 += tree69(f)
	s0 += tree72(f)
	s0 += tree75(f)
	s0 += tree78(f)
	s0 += tree81(f)
	s0 += tree84(f)
	s0 += tree87(f)
	s1 += tree1(f)
	s1 += tree4(f)
	s1 += tree7(f)
	s1 += tree10(f)
	s1 += tree13(f)
	s1 += tree16(f)
	s1 += tree19(f)
	s1 += tree22(f)
	s1 += tree25(f)
	s1 += tree28(f)
	s1 += tree31(f)
	s1 += tree34(f)
	s1 += tree37(f)
	s1 += tree40(f)
	s1 += tree43(f)
	s1 += tree46(f)
	s1 += tree49(f)
	s1 += tree52(f)
	s1 += tree55(f)
	s1 += tree58(f)
	s1 += tree61(f)
	s1 += tree64(f)
	s1 += tree67(f)
	s1 += tree70(f)
	s1 += tree73(f)
	s1 += tree76(f)
	s1 += tree79(f)
	s1 += tree82(f)
	s1 += tree85(f)
	s1 += tree88(f)
	s2 += tree2(f)
	s2 += tree5(f)
	s2 += tree8(f)
	s2 += tree11(f)
	s2 += tree14(f)
	s2 += tree17(f)
	s2 += tree20(f)
	s2 += tree23(f)
	s2 += tree26(f)
	s2 += tree29(f)
	s2 += tree32(f)
	s2 += tree35(f)
	s2 += tree38(f)
	s2 += tree41(f)
	s2 += tree44(f)
	s2 += tree47(f)
	s2 += tree50(f)
	s2 += tree53(f)
	s2 += tree56(f)
	s2 += tree59(f)
	s2 += tree62(f)
	s2 += tree65(f)
	s2 += tree68(f)
	s2 += tree71(f)
	s2 += tree74(f)
	s2 += tree77(f)
	s2 += tree80(f)
	s2 += tree83(f)
	s2 += tree86(f)
	s2 += tree89(f)
	return []float64{s0, s1, s2}
}

// Predict returns softmax probabilities over all classes.
func Predict(f []float64) []float64 {
	s := PredictScores(f)
	max := s[0]
	for _, v := range s {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for i := range s {
		s[i] = math.Exp(s[i] - max)
		sum += s[i]
	}
	for i := range s {
		s[i] /= sum
	}
	return s
}

// PredictClass returns the class index with the highest probability.
func PredictClass(f []float64) int {
	p := PredictScores(f)
	best := 0
	for i := range p {
		if p[i] > p[best] {
			best = i
		}
	}
	return best
}

func tree0(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.048871864000000001
			}
			return 0.10076444599999999
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.070719703999999994
		}
		return -0.019344050000000002
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.067467319999999997
	}
	if float32(f[1]) < 1.12861967 {
		return 0.13173823000000001
	}
	return -0.031093273000000001
}

func tree1(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.071993314000000003
		}
		return 0.058672009999999997
	}
	if float32(f[1]) < -0.0475930758 {
		if float32(f[7]) < -2.64903045 {
			return 0.097814600000000002
		}
		return -0.038204084999999999
	}
	if float32(f[3]) < -1.67684293 {
		return 0.044485944999999999
	}
	return 0.12323667000000001
}

func tree2(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[0]) < 1.05086458 {
			if float32(f[6]) < 1.2293998 {
				return 0.1286207
			}
			return -0.052063643999999999
		}
		if float32(f[2]) < -1.98869884 {
			return 0.026631404000000001
		}
		return -0.068690285000000004
	}
	if float32(f[6]) < -1.6818099 {
		if float32(f[2]) < 0.948501527 {
			return 0.13028393999999999
		}
		return -0.043163623999999998
	}
	if float32(f[6]) < -0.307667077 {
		return -0.0354183
	}
	return -0.072515780000000002
}

func tree3(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.046448454
			}
			return 0.090708869999999997
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.068340200000000004
		}
		return -0.018345690000000001
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.065302810000000003
	}
	if float32(f[1]) < 1.12861967 {
		return 0.11542909599999999
	}
	return -0.029147467
}

func tree4(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.069201040000000005
		}
		return 0.054184999999999997
	}
	if float32(f[1]) < -0.968803227 {
		if float32(f[3]) < 0.724808931 {
			return -0.053532827999999998
		}
		return 0.089640789999999998
	}
	if float32(f[0]) < 1.05086458 {
		return 0.062226173000000003
	}
	return 0.11939212
}

func tree5(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[0]) < 1.05086458 {
			if float32(f[6]) < 1.2293998 {
				return 0.11343997
			}
			return -0.050474018000000002
		}
		if float32(f[2]) < -1.98869884 {
			return 0.026200343000000001
		}
		return -0.066787020000000002
	}
	if float32(f[6]) < -1.6818099 {
		if float32(f[2]) < 0.948501527 {
			return 0.11699748
		}
		return -0.041636279999999998
	}
	if float32(f[6]) < -0.307667077 {
		return -0.033401802000000001
	}
	return -0.069868379999999994
}

func tree6(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.044142883000000001
			}
			return 0.082137310000000005
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.066215079999999996
		}
		return -0.017116270999999999
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.063039380000000006
	}
	if float32(f[1]) < 1.12861967 {
		return 0.10294657
	}
	return -0.028034052
}

func tree7(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.066722593999999996
		}
		return 0.050199556999999999
	}
	if float32(f[1]) < 0.0596084259 {
		if float32(f[0]) < 1.48477829 {
			return -0.020656131000000001
		}
		return 0.095067344999999998
	}
	if float32(f[3]) < -1.67684293 {
		return 0.033517007000000001
	}
	return 0.10096586
}

func tree8(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[2]) < -2.05602145 {
				return 0.012548487000000001
			}
			return -0.058246600000000003
		}
		if float32(f[6]) < 1.2293998 {
			return 0.10396933
		}
		return -0.048963036000000001
	}
	if float32(f[6]) < -1.6818099 {
		if float32(f[2]) < 0.948501527 {
			return 0.10664058
		}
		return -0.04015523
	}
	if float32(f[6]) < -0.307667077 {
		return -0.031588150000000002
	}
	return -0.067641000000000007
}

func tree9(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.04194531
			}
			return 0.075047836000000007
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.064331509999999995
		}
		return -0.016004576999999999
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.061021876000000003
	}
	if float32(f[1]) < 1.12861967 {
		return 0.093184340000000004
	}
	return -0.026144067
}

func tree10(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.064529089999999997
		}
		return 0.046635505000000001
	}
	if float32(f[1]) < -0.968803227 {
		if float32(f[3]) < 0.724808931 {
			return -0.050705235000000001
		}
		return 0.084231793999999999
	}
	if float32(f[0]) < 1.05086458 {
		return 0.050575000000000002
	}
	return 0.098628889999999997
}

func tree11(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[0]) < 1.05086458 {
			if float32(f[6]) < 1.2293998 {
				return 0.093374879999999993
			}
			return -0.047451403000000003
		}
		if float32(f[2]) < -1.98869884 {
			return 0.025313415999999998
		}
		return -0.063526399999999997
	}
	if float32(f[6]) < -1.6818099 {
		if float32(f[2]) < 0.948501527 {
			return 0.098399219999999996
		}
		return -0.038725722999999997
	}
	if float32(f[6]) < -0.307667077 {
		return -0.029708544
	}
	return -0.06556505
}

func tree12(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.039773572
			}
			return 0.069054104000000005
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.062605859999999999
		}
		return -0.014759306999999999
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.059191037000000002
	}
	if float32(f[1]) < 1.12861967 {
		return 0.085260009999999997
	}
	return -0.025062649999999999
}

func tree13(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.062532180000000007
		}
		return 0.043431024999999998
	}
	if float32(f[1]) < 0.0596084259 {
		if float32(f[7]) < -2.64903045 {
			return 0.072965199999999994
		}
		return -0.027455879999999998
	}
	if float32(f[3]) < -1.67684293 {
		return 0.025585696000000002
	}
	return 0.085253074999999998
}

func tree14(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[6]) < 0.214168921 {
				return -0.064322519999999994
			}
			return -0.0022027022
		}
		if float32(f[6]) < 1.2293998 {
			return 0.087991200000000006
		}
		return -0.046051309999999998
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.093897350000000004
		}
		return -0.045960872999999999
	}
	if float32(f[6]) < -0.307667077 {
		return -0.031518847000000003
	}
	return -0.063772770000000006
}

func tree15(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.037680591999999999
			}
			return 0.063982873999999995
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.061027177000000002
		}
		return -0.013660932000000001
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.057620490000000003
	}
	if float32(f[1]) < 1.12861967 {
		return 0.078757830000000001
	}
	return -0.023266213000000001
}

func tree16(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.060733664999999999
		}
		return 0.040532110000000003
	}
	if float32(f[1]) < -0.0475930758 {
		if float32(f[7]) < -2.64903045 {
			return 0.06771887
		}
		return -0.034490390000000003
	}
	if float32(f[3]) < -1.67684293 {
		return 0.023536688
	}
	return 0.077532370000000003
}

func tree17(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[6]) < 0.214168921 {
				return -0.063011600000000001
			}
			return -0.0013309558
		}
		if float32(f[6]) < 1.2293998 {
			return 0.081537954999999995
		}
		return -0.044676285000000003
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.088066119999999998
		}
		return -0.044360669999999998
	}
	if float32(f[6]) < -0.307667077 {
		return -0.029556101000000001
	}
	return -0.062059212000000002
}

func tree18(f []float64) float64 {
	if float32(f[2]) < 0.0529755503 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.892246604 {
				return -0.035666086
			}
			return 0.059551984000000002
		}
		if float32(f[2]) < -0.0331099592 {
			return -0.059573907000000002
		}
		return -0.01262595
	}
	if float32(f[3]) < -0.0593615547 {
		return -0.056211628
	}
	if float32(f[1]) < 1.65182078 {
		return 0.071938950000000002
	}
	return -0.040869620000000002
}

func tree19(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.059095870000000002
		}
		return 0.037897027999999999
	}
	if float32(f[0]) < 1.05086458 {
		if float32(f[3]) < 0.742841184 {
			return -0.031519896999999998
		}
		return 0.073374889999999998
	}
	if float32(f[1]) < -0.976249278 {
		return 0.026065936000000001
	}
	return 0.081106834000000003
}

func tree20(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[0]) < 1.05086458 {
			if float32(f[6]) < 1.2293998 {
				return 0.075289079999999994
			}
			return -0.043370159999999998
		}
		if float32(f[2]) < -1.98869884 {
			return 0.026877346999999999
		}
		return -0.060753834
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.08323324
		}
		return -0.042839512000000003
	}
	if float32(f[6]) < -0.307667077 {
		return -0.027679427
	}
	return -0.060507434999999998
}

func tree21(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.892246604 {
				return -0.049984870000000001
			}
			return 0.049452210000000003
		}
		return -0.060478400000000002
	}
	if float32(f[3]) < -0.50808543 {
		if float32(f[5]) < -2.66146016 {
			return 0.016288754999999999
		}
		return -0.055816032000000002
	}
	if float32(f[1]) < 1.12861967 {
		return 0.063987660000000002
	}
	return -0.036424126000000001
}

func tree22(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.057574075000000002
		}
		return 0.035479723999999997
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.859681129 {
			return -0.056428112000000002
		}
		return 0.068969240000000001
	}
	if float32(f[6]) < 0.214168921 {
		return 0.087376549999999997
	}
	return 0.050829436999999998
}

func tree23(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[6]) < 0.214168921 {
				return -0.0620251
			}
			return 0.00022224351
		}
		if float32(f[6]) < 1.2293998 {
			return 0.072196090000000004
		}
		return -0.042117047999999997
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.07917457
		}
		return -0.041409474000000002
	}
	if float32(f[6]) < -0.307667077 {
		return -0.026359054999999999
	}
	return -0.059084959999999999
}

func tree24(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.892246604 {
				return -0.048190183999999997
			}
			return 0.046493657000000001
		}
		return -0.059362183999999998
	}
	if float32(f[3]) < -0.50808543 {
		if float32(f[5]) < -2.66146016 {
			return 0.015920569999999998
		}
		return -0.054375380000000001
	}
	if float32(f[1]) < 1.12861967 {
		return 0.05987203
	}
	return -0.034683763999999999
}

func tree25(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			return -0.056176549999999999
		}
		return 0.033302659999999998
	}
	if float32(f[0]) < 1.05086458 {
		if float32(f[3]) < 0.120538771 {
			return -0.046515296999999997
		}
		return 0.054724085999999998
	}
	if float32(f[2]) < -1.98869884 {
		return 0.037285293999999997
	}
	return 0.076291250000000005
}

func tree26(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[6]) < 0.214168921 {
				return -0.060187607999999997
			}
			return 0.0011478815000000001
		}
		if float32(f[6]) < 1.2293998 {
			return 0.068118150000000002
		}
		return -0.040913194
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.075886980000000007
		}
		return -0.040090106
	}
	if float32(f[6]) < -0.307667077 {
		return -0.024687503999999999
	}
	return -0.057823951999999998
}

func tree27(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.892246604 {
				return -0.046490367999999997
			}
			return 0.043772772000000001
		}
		return -0.058333874000000001
	}
	if float32(f[3]) < -0.50808543 {
		if float32(f[5]) < -2.66146016 {
			return 0.015695099000000001
		}
		return -0.053083855999999999
	}
	if float32(f[7]) < 1.43779933 {
		return 0.055903118000000002
	}
	return -0.040430404000000003
}

func tree28(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.041127067000000003
			}
			return -0.058759909999999999
		}
		return 0.031291615000000002
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.859681129 {
			return -0.054675309999999998
		}
		return 0.061603110000000003
	}
	if float32(f[6]) < 0.214168921 {
		return 0.079976850000000002
	}
	return 0.04481562
}

func tree29(f []float64) float64 {
	if float32(f[3]) < -0.55916369 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[1]) < 1.53348327 {
				return 0.064479449999999994
			}
			return -0.027634882999999999
		}
		return -0.05600637
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.073362880000000005
		}
		return -0.038833257000000003
	}
	if float32(f[6]) < -0.198439032 {
		return -0.016895140999999999
	}
	return -0.057133074999999998
}

func tree30(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[2]) < -0.927492857 {
				return -0.03025197
			}
			return 0.044464216000000001
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.027384868
		}
		return -0.059568285999999998
	}
	if float32(f[3]) < -0.50808543 {
		return -0.039325986
	}
	return 0.071995480000000001
}

func tree31(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.039495725000000002
			}
			return -0.057732819999999997
		}
		return 0.029389578999999999
	}
	if float32(f[0]) < 1.48477829 {
		if float32(f[3]) < 0.742841184 {
			return -0.013492550000000001
		}
		return 0.060319709999999999
	}
	if float32(f[6]) < 3.00943089 {
		return 0.073435479999999997
	}
	return 0.019804148000000001
}

func tree32(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[6]) < 0.214168921 {
				return -0.058150735000000002
			}
			return 0.0027277811999999999
		}
		if float32(f[6]) < 1.2293998 {
			return 0.061936449999999997
		}
		return -0.039724137999999999
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.070188449999999999
		}
		return -0.037861746000000002
	}
	if float32(f[6]) < -0.307667077 {
		return -0.021776576999999998
	}
	return -0.055533987
}

func tree33(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[2]) < -0.927492857 {
				return -0.028551984999999998
			}
			return 0.041738536
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.026076375999999998
		}
		return -0.05854289
	}
	if float32(f[3]) < -0.50808543 {
		return -0.038147762000000002
	}
	if float32(f[1]) < -1.02967799 {
		return 0.040838882
	}
	return 0.074149065
}

func tree34(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.037696935000000001
			}
			return -0.056811746000000003
		}
		return 0.027680640999999999
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.859681129 {
			return -0.053339640000000001
		}
		return 0.055387515999999998
	}
	if float32(f[5]) < -2.10055065 {
		return 0.02940309
	}
	return 0.065080344999999998
}

func tree35(f []float64) float64 {
	if float32(f[3]) < -0.55916369 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[1]) < 1.53348327 {
				return 0.059152999999999997
			}
			return -0.027638434999999999
		}
		return -0.053571003999999998
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.067614019999999997
		}
		return -0.036436240000000002
	}
	if float32(f[6]) < -0.198439032 {
		return -0.014619762
	}
	return -0.055040146999999998
}

func tree36(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[2]) < -0.927492857 {
				return -0.026665899999999999
			}
			return 0.039261999999999998
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.024797778999999999
		}
		return -0.05758663
	}
	if float32(f[3]) < -0.50808543 {
		return -0.037180945
	}
	if float32(f[1]) < -1.02967799 {
		return 0.038408014999999997
	}
	return 0.071769799999999995
}

func tree37(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.035953060000000002
			}
			return -0.055963289999999999
		}
		return 0.026127595
	}
	if float32(f[0]) < 1.05086458 {
		if float32(f[3]) < 0.120538771 {
			return -0.044130693999999998
		}
		return 0.043830910000000001
	}
	if float32(f[2]) < -1.98869884 {
		return 0.027626787999999999
	}
	return 0.065356195000000006
}

func tree38(f []float64) float64 {
	if float32(f[3]) < -0.55916369 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[9]) < -2.97147298 {
				return -0.041677029999999997
			}
			return 0.05552381
		}
		return -0.052169103000000001
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.06507665
		}
		return -0.035004995999999997
	}
	if float32(f[6]) < -0.198439032 {
		return -0.013425220999999999
	}
	return -0.054115391999999998
}

func tree39(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.050838687
			}
			return 0.032529384000000001
		}
		return -0.055344373000000002
	}
	if float32(f[3]) < -1.22425401 {
		return -0.049321570000000002
	}
	if float32(f[7]) < 1.43779933 {
		return 0.047361907000000002
	}
	return -0.037484653
}

func tree40(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.034235175999999999
			}
			return -0.055192287999999999
		}
		return 0.024686128000000002
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.742841184 {
			return -0.057103552000000002
		}
		return 0.046980335999999998
	}
	if float32(f[6]) < 0.214168921 {
		return 0.070508730000000006
	}
	return 0.034325649999999999
}

func tree41(f []float64) float64 {
	if float32(f[3]) < -0.338255197 {
		if float32(f[9]) < -1.75056386 {
			if float32(f[6]) < 0.214168921 {
				return -0.056304689999999998
			}
			return 0.0045499457000000004
		}
		if float32(f[6]) < 1.2293998 {
			return 0.055816468000000001
		}
		return -0.039736103000000002
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.062432356000000001
		}
		return -0.033627845000000003
	}
	if float32(f[6]) < 0.323042125 {
		return -0.023779939999999999
	}
	return -0.055622119999999997
}

func tree42(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[6]) < 0.430280566 {
				return -0.022000974
			}
			return 0.039440375
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.02287985
		}
		return -0.056255050000000001
	}
	if float32(f[3]) < -0.50808543 {
		return -0.034833179999999998
	}
	if float32(f[4]) < -0.675708055 {
		return 0.032461200000000003
	}
	return 0.068281759999999997
}

func tree43(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.032672609999999998
			}
			return -0.054458517999999997
		}
		return 0.02321117
	}
	if float32(f[0]) < 1.48477829 {
		if float32(f[3]) < 0.742841184 {
			return -0.014874155999999999
		}
		return 0.050242644000000003
	}
	if float32(f[6]) < 3.00943089 {
		return 0.064588110000000004
	}
	return 0.014231229
}

func tree44(f []float64) float64 {
	if float32(f[3]) < -0.55916369 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[1]) < 1.53348327 {
				return 0.0530319
			}
			return -0.027643535
		}
		return -0.050225943000000002
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.061021984000000001
		}
		return -0.032821573
	}
	if float32(f[6]) < -0.198439032 {
		return -0.011261738
	}
	return -0.052383135999999997
}

func tree45(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[3]) < -1.20291007 {
				return -0.035732399999999997
			}
			return 0.029609536999999998
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.021736292000000001
		}
		return -0.055491595999999997
	}
	if float32(f[3]) < -0.50808543 {
		return -0.034000304000000002
	}
	if float32(f[1]) < -1.02967799 {
		return 0.030858635999999998
	}
	return 0.06712601
}

func tree46(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.031049673999999999
			}
			return -0.053796330000000003
		}
		return 0.021960989
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.859681129 {
			return -0.051542974999999998
		}
		return 0.046062770000000003
	}
	if float32(f[5]) < -2.10055065 {
		return 0.020435867999999999
	}
	return 0.057558294000000003
}

func tree47(f []float64) float64 {
	if float32(f[3]) < -0.55916369 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[1]) < 1.53348327 {
				return 0.050913225999999999
			}
			return -0.02747761
		}
		return -0.048826410000000001
	}
	if float32(f[6]) < -1.60694182 {
		if float32(f[2]) < 0.948501527 {
			return 0.059064683
		}
		return -0.031616386000000003
	}
	if float32(f[6]) < -0.198439032 {
		return -0.010602243000000001
	}
	return -0.051567535999999997
}

func tree48(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.048423941999999998
			}
			return 0.030503450000000001
		}
		return -0.053676086999999997
	}
	if float32(f[3]) < -1.22425401 {
		return -0.046511860000000002
	}
	if float32(f[7]) < 1.43779933 {
		return 0.042874240000000001
	}
	return -0.033825019999999997
}

func tree49(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.029377183000000001
			}
			return -0.053175251999999999
		}
		return 0.020798069999999998
	}
	if float32(f[0]) < 1.48477829 {
		if float32(f[3]) < 0.742841184 {
			return -0.015095024
		}
		return 0.046405309999999998
	}
	if float32(f[6]) < 3.00943089 {
		return 0.061276431999999999
	}
	return 0.01304043
}

func tree50(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[1]) < 1.53348327 {
				return 0.056976396999999998
			}
			return -0.01304396
		}
		return -0.04403555
	}
	if float32(f[6]) < -1.32554853 {
		if float32(f[2]) < 0.948501527 {
			return 0.058711952999999997
		}
		return -0.033436413999999998
	}
	if float32(f[6]) < 0.430280566 {
		return -0.012268237
	}
	return -0.054394748
}

func tree51(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[6]) < 0.430280566 {
				return -0.019945376000000001
			}
			return 0.035394769999999999
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.019983580000000001
		}
		return -0.054421990000000003
	}
	if float32(f[3]) < -0.50808543 {
		return -0.032452920000000003
	}
	if float32(f[1]) < -1.02967799 {
		return 0.026663541999999998
	}
	return 0.064943500000000001
}

func tree52(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.027801288
			}
			return -0.052560906999999997
		}
		return 0.019691490999999998
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.859681129 {
			return -0.050647615999999999
		}
		return 0.042372226999999998
	}
	if float32(f[6]) < 0.214168921 {
		return 0.065027080000000001
	}
	return 0.026687576000000001
}

func tree53(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[9]) < -2.97147298 {
				return -0.038108113999999998
			}
			return 0.056032967000000003
		}
		return -0.042736202000000001
	}
	if float32(f[6]) < -1.32554853 {
		if float32(f[2]) < 0.948501527 {
			return 0.05733916
		}
		return -0.032622169999999999
	}
	if float32(f[6]) < 0.430280566 {
		return -0.011581434999999999
	}
	return -0.053741004000000002
}

func tree54(f []float64) float64 {
	if float32(f[2]) < 1.00508904 {
		if float32(f[1]) < -0.714657485 {
			if float32(f[2]) < -0.927492857 {
				return -0.021092085
			}
			return 0.031759429999999998
		}
		if float32(f[1]) < 0.000418641197 {
			return -0.018971321999999999
		}
		return -0.053773399999999999
	}
	if float32(f[3]) < -0.50808543 {
		return -0.031454551999999997
	}
	if float32(f[4]) < -0.675708055 {
		return 0.023945238000000001
	}
	return 0.063558206000000006
}

func tree55(f []float64) float64 {
	if float32(f[0]) < -0.111699365 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[0]) < -1.88413751 {
				return -0.026189570999999998
			}
			return -0.052014900000000003
		}
		return 0.018772831
	}
	if float32(f[0]) < 1.48477829 {
		if float32(f[3]) < 0.742841184 {
			return -0.015178960999999999
		}
		return 0.043244234999999999
	}
	if float32(f[6]) < 3.00943089 {
		return 0.058572399999999997
	}
	return 0.012178606
}

func tree56(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[1]) < 1.53348327 {
				return 0.053971419999999999
			}
			return -0.01402463
		}
		return -0.041515957999999999
	}
	if float32(f[6]) < -1.32554853 {
		if float32(f[2]) < 0.948501527 {
			return 0.055903840000000003
		}
		return -0.031356584
	}
	if float32(f[6]) < 0.430280566 {
		return -0.010751189
	}
	return -0.053202386999999997
}

func tree57(f []float64) float64 {
	if float32(f[2]) < 1.15687466 {
		if float32(f[1]) < -0.50107795 {
			if float32(f[6]) < 0.430280566 {
				return -0.019383483999999999
			}
			return 0.033045555999999997
		}
		if float32(f[6]) < 2.33391738 {
			return -0.048747405000000001
		}
		return -0.0047735036000000003
	}
	if float32(f[3]) < 1.46996367 {
		if float32(f[5]) < -0.794712543 {
			return 0.041652403999999997
		}
		return -0.030771126999999999
	}
	return 0.062418340000000003
}

func tree58(f []float64) float64 {
	if float32(f[0]) < -0.0115920445 {
		if float32(f[3]) < 3.51350212 {
			if float32(f[1]) < 1.52481306 {
				return -0.04575717
			}
			return 0.004026282
		}
		return 0.024934629999999999
	}
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.859681129 {
			return -0.046213259999999999
		}
		return 0.036392665999999997
	}
	if float32(f[6]) < 0.214168921 {
		return 0.062732739999999995
	}
	return 0.023178542
}

func tree59(f []float64) float64 {
	if float32(f[3]) < 0.120538771 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[6]) < 1.53871822 {
				return 0.038201289999999999
			}
			return -0.042259637000000003
		}
		return -0.046596350000000002
	}
	if float32(f[6]) < -1.6818099 {
		if float32(f[2]) < 0.756014764 {
			return 0.048684638000000002
		}
		return 0.0015243021
	}
	if float32(f[3]) < 0.742841184 {
		return -0.023375360000000001
	}
	return -0.052439975999999999
}

func tree60(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.045647487
			}
			return 0.027543756999999999
		}
		return -0.051930863000000001
	}
	if float32(f[3]) < -1.22425401 {
		return -0.043149035000000002
	}
	if float32(f[5]) < -1.73576534 {
		return 0.067379460000000002
	}
	return 0.017555411999999999
}

func tree61(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.052606291999999999
			}
			return 0.022956450999999999
		}
		if float32(f[2]) < 1.77441013 {
			return 0.055224843000000003
		}
		return -0.056234310000000003
	}
	if float32(f[5]) < -2.10055065 {
		if float32(f[9]) < -2.66635728 {
			return 0.050621329999999999
		}
		return -0.028168568000000001
	}
	if float32(f[3]) < -1.65443826 {
		return 0.0083198170000000002
	}
	return 0.060308736000000002
}

func tree62(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[0]) < 0.987625062 {
			if float32(f[6]) < 0.892246604 {
				return 0.057350505000000003
			}
			return -0.00016750865
		}
		if float32(f[2]) < -1.73653746 {
			return 0.037348893000000001
		}
		return -0.045041657999999998
	}
	if float32(f[6]) < 0.430280566 {
		if float32(f[2]) < 0.948501527 {
			return 0.025821809000000001
		}
		return -0.044250138000000001
	}
	return -0.05227996
}

func tree63(f []float64) float64 {
	if float32(f[2]) < 1.77441013 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.188579872 {
				return -0.024810080000000002
			}
			return 0.040670257000000001
		}
		if float32(f[5]) < -2.39211297 {
			return 0.0063097285999999999
		}
		return -0.045396810000000003
	}
	if float32(f[3]) < 1.60219622 {
		return 0.015313037
	}
	return 0.060385800000000003
}

func tree64(f []float64) float64 {
	if float32(f[0]) < 0.472140968 {
		if float32(f[3]) < 1.40423346 {
			if float32(f[1]) < 2.01135683 {
				return -0.0504584
			}
			return 0.015299975
		}
		if float32(f[2]) < 1.77441013 {
			return 0.051398157999999999
		}
		return -0.049700926999999999
	}
	if float32(f[5]) < -2.10055065 {
		if float32(f[5]) < -2.34273267 {
			return 0.032270376000000003
		}
		return -0.054925359999999999
	}
	if float32(f[3]) < -1.65443826 {
		return 0.0072258556999999996
	}
	return 0.052437490000000003
}

func tree65(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[9]) < -2.719733 {
			return -0.042782027
		}
		if float32(f[0]) < 1.48477829 {
			return 0.051549483
		}
		return -0.031986899999999999
	}
	if float32(f[6]) < -0.198439032 {
		if float32(f[2]) < 0.756014764 {
			return 0.034624264000000002
		}
		return -0.034654886000000003
	}
	if float32(f[6]) < 0.430280566 {
		return -0.016476636999999999
	}
	return -0.051713790000000003
}

func tree66(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.044026296999999999
			}
			return 0.025094498
		}
		return -0.050983198
	}
	if float32(f[3]) < -1.22425401 {
		return -0.041616569999999999
	}
	if float32(f[5]) < -1.73576534 {
		return 0.065043050000000005
	}
	return 0.015309191999999999
}

func tree67(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.051273763
			}
			return 0.021172263
		}
		if float32(f[2]) < 1.77441013 {
			return 0.052404642000000001
		}
		return -0.054033989999999997
	}
	if float32(f[6]) < 0.214168921 {
		return 0.059304981999999999
	}
	if float32(f[3]) < -0.66300571 {
		return -0.031632214999999998
	}
	return 0.044599769999999997
}

func tree68(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[0]) < 0.987625062 {
			if float32(f[6]) < 0.892246604 {
				return 0.055418559999999999
			}
			return -0.0010088694
		}
		if float32(f[2]) < -1.73653746 {
			return 0.03534963
		}
		return -0.043450694999999998
	}
	if float32(f[6]) < -1.32554853 {
		if float32(f[2]) < 0.756014764 {
			return 0.050791215000000001
		}
		return -0.0085757379999999994
	}
	if float32(f[6]) < 0.430280566 {
		return -0.010082064
	}
	return -0.051179130000000003
}

func tree69(f []float64) float64 {
	if float32(f[2]) < 1.77441013 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.188579872 {
				return -0.023065576000000001
			}
			return 0.038161527000000001
		}
		if float32(f[5]) < -2.39211297 {
			return 0.0064107477999999999
		}
		return -0.044415870000000003
	}
	return 0.055432166999999997
}

func tree70(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.050597160000000002
			}
			return 0.020221495999999999
		}
		if float32(f[2]) < 1.77441013 {
			return 0.051106970000000002
		}
		return -0.053042213999999997
	}
	if float32(f[6]) < 0.214168921 {
		return 0.058002587000000001
	}
	if float32(f[3]) < -0.66300571 {
		return -0.030881057
	}
	return 0.043238483000000001
}

func tree71(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[9]) < -2.719733 {
			return -0.041092799999999999
		}
		if float32(f[0]) < 1.48477829 {
			return 0.049330819999999997
		}
		return -0.030127272
	}
	if float32(f[6]) < 0.430280566 {
		if float32(f[2]) < 0.948501527 {
			return 0.023456823000000002
		}
		return -0.041910603999999997
	}
	return -0.050674695999999998
}

func tree72(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.042504420000000001
			}
			return 0.023059273000000002
		}
		return -0.050096173000000001
	}
	if float32(f[5]) < -1.90349066 {
		if float32(f[9]) < -2.91487789 {
			return 0.010665519
		}
		return 0.063467875000000007
	}
	if float32(f[2]) < 1.77441013 {
		return -0.013995933
	}
	return 0.049480807000000002
}

func tree73(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.04994345
			}
			return 0.019528953000000002
		}
		if float32(f[2]) < 1.77441013 {
			return 0.049627468000000001
		}
		return -0.051843893000000002
	}
	if float32(f[5]) < -2.10055065 {
		if float32(f[9]) < -2.66635728 {
			return 0.046296459999999998
		}
		return -0.028540260000000001
	}
	if float32(f[3]) < -1.65443826 {
		return 0.0054628732999999997
	}
	return 0.056536524999999997
}

func tree74(f []float64) float64 {
	if float32(f[3]) < -1.56738353 {
		if float32(f[9]) < -2.719733 {
			return -0.036032679999999997
		}
		if float32(f[0]) < 1.32191753 {
			return 0.054649024999999997
		}
		return -0.00010176732
	}
	if float32(f[6]) < -1.32554853 {
		if float32(f[2]) < 0.756014764 {
			return 0.050178300000000002
		}
		return -0.0070736879999999999
	}
	if float32(f[6]) < 0.559685588 {
		return -0.0083951979999999995
	}
	return -0.050668640000000001
}

func tree75(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.041626080000000003
			}
			return 0.022002649999999999
		}
		return -0.049638613999999998
	}
	if float32(f[5]) < -1.90349066 {
		if float32(f[7]) < -2.30626488 {
			return 0.0090924209999999998
		}
		return 0.061453313000000002
	}
	if float32(f[2]) < 1.77441013 {
		return -0.01365443
	}
	return 0.048215042999999999
}

func tree76(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.23191047 {
				return -0.048411757
			}
			return 0.026620774
		}
		if float32(f[2]) < 1.77441013 {
			return 0.048146803000000002
		}
		return -0.050622414999999997
	}
	if float32(f[6]) < 0.214168921 {
		return 0.056099385000000002
	}
	if float32(f[3]) < -0.66300571 {
		return -0.030103656999999999
	}
	return 0.041052945
}

func tree77(f []float64) float64 {
	if float32(f[3]) < -1.21869707 {
		if float32(f[9]) < -2.719733 {
			return -0.039029516
		}
		if float32(f[0]) < 1.48477829 {
			return 0.047473482999999997
		}
		return -0.028899536
	}
	if float32(f[6]) < -0.198439032 {
		if float32(f[2]) < 0.756014764 {
			return 0.030843347
		}
		return -0.031041255
	}
	if float32(f[5]) < -1.5155437 {
		return -0.020265970000000001
	}
	return -0.054346323000000002
}

func tree78(f []float64) float64 {
	if float32(f[2]) < 1.77441013 {
		if float32(f[1]) < -1.08933675 {
			if float32(f[6]) < 0.188579872 {
				return -0.020826686000000001
			}
			return 0.035349466000000003
		}
		if float32(f[5]) < -2.39211297 {
			return 0.0067524569999999999
		}
		return -0.042670283000000003
	}
	if float32(f[3]) < 1.60947299 {
		return 0.015071769
	}
	return 0.054260608000000002
}

func tree79(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.048632032999999998
			}
			return 0.018371364000000001
		}
		if float32(f[2]) < 1.77441013 {
			return 0.046587522999999999
		}
		return -0.04943612
	}
	if float32(f[5]) < -2.10055065 {
		if float32(f[9]) < -2.66635728 {
			return 0.044308239999999999
		}
		return -0.028594656
	}
	if float32(f[3]) < -1.65443826 {
		return 0.0058181235000000003
	}
	return 0.054827626999999997
}

func tree80(f []float64) float64 {
	if float32(f[3]) < -1.56738353 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[9]) < -2.60954881 {
				return -0.010371807
			}
			return 0.054038919999999997
		}
		return -0.029204219999999999
	}
	if float32(f[6]) < 0.559685588 {
		if float32(f[2]) < 0.948501527 {
			return 0.020496845
		}
		return -0.038971635999999997
	}
	return -0.049860460000000002
}

func tree81(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.040126250000000002
			}
			return 0.020046955000000002
		}
		return -0.048817265999999998
	}
	if float32(f[5]) < -1.90349066 {
		if float32(f[9]) < -2.91487789 {
			return 0.0086227270000000002
		}
		return 0.060037005999999997
	}
	if float32(f[9]) < 0.912406504 {
		return -0.013818047999999999
	}
	return 0.045519568000000003
}

func tree82(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.047985519999999997
			}
			return 0.017815576999999999
		}
		if float32(f[2]) < 1.77441013 {
			return 0.045192759999999998
		}
		return -0.048256042999999998
	}
	if float32(f[6]) < 0.214168921 {
		return 0.054486960000000001
	}
	if float32(f[3]) < -0.66300571 {
		return -0.029292631999999999
	}
	return 0.039165165000000002
}

func tree83(f []float64) float64 {
	if float32(f[3]) < -1.56738353 {
		if float32(f[0]) < 1.48477829 {
			if float32(f[9]) < -2.60954881 {
				return -0.010010391
			}
			return 0.053276366999999998
		}
		return -0.028642919999999999
	}
	if float32(f[6]) < -1.32554853 {
		if float32(f[2]) < 0.756014764 {
			return 0.048167557
		}
		return -0.0043687639999999998
	}
	if float32(f[0]) < -1.25450051 {
		return -0.058132099999999999
	}
	return -0.011075313
}

func tree84(f []float64) float64 {
	if float32(f[2]) < 1.77441013 {
		if float32(f[1]) < -1.29316437 {
			if float32(f[7]) < -1.80681491 {
				return -0.027402838999999998
			}
			return 0.036935142999999997
		}
		if float32(f[5]) < -2.16639376 {
			return 0.008841837
		}
		return -0.044524874999999998
	}
	return 0.049475867
}

func tree85(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 1.28823042 {
			if float32(f[1]) < 2.01135683 {
				return -0.047360766999999998
			}
			return 0.017310902
		}
		if float32(f[2]) < 1.77441013 {
			return 0.044101179999999997
		}
		return -0.047311264999999998
	}
	if float32(f[5]) < -2.10055065 {
		if float32(f[9]) < -2.66635728 {
			return 0.042750365999999998
		}
		return -0.028008534000000002
	}
	if float32(f[3]) < -1.65443826 {
		return 0.0057842850000000001
	}
	return 0.053202636999999997
}

func tree86(f []float64) float64 {
	if float32(f[6]) < 0.994292617 {
		if float32(f[2]) < 0.948501527 {
			if float32(f[9]) < 1.07584345 {
				return 0.02984264
			}
			return -0.05543841
		}
		if float32(f[3]) < -0.50808543 {
			return -0.0085244689999999998
		}
		return -0.044906090000000003
	}
	if float32(f[3]) < -1.65443826 {
		return 0.013685609
	}
	return -0.048267465000000002
}

func tree87(f []float64) float64 {
	if float32(f[2]) < -0.483758569 {
		if float32(f[1]) < -1.24732506 {
			if float32(f[6]) < 0.188579872 {
				return -0.039169993
			}
			return 0.019495556000000001
		}
		return -0.048044360000000001
	}
	if float32(f[5]) < -1.90349066 {
		if float32(f[9]) < -2.91487789 {
			return 0.0082865230000000005
		}
		return 0.057533837999999997
	}
	if float32(f[9]) < 0.912406504 {
		return -0.013906828
	}
	return 0.043440510000000002
}

func tree88(f []float64) float64 {
	if float32(f[0]) < 0.784399033 {
		if float32(f[3]) < 0.256433368 {
			return -0.051730619999999998
		}
		if float32(f[2]) < 0.688973606 {
			return 0.027825061000000002
		}
		return -0.033515445999999997
	}
	if float32(f[6]) < 0.214168921 {
		return 0.052968670000000002
	}
	if float32(f[3]) < -0.66300571 {
		return -0.028383453999999999
	}
	return 0.037239260000000003
}

func tree89(f []float64) float64 {
	if float32(f[6]) < 0.994292617 {
		if float32(f[2]) < 0.948501527 {
			if float32(f[9]) < 1.07584345 {
				return 0.028546723999999999
			}
			return -0.054513970000000002
		}
		if float32(f[3]) < -0.50808543 {
			return -0.0081571999999999999
		}
		return -0.04422077
	}
	if float32(f[3]) < -1.65443826 {
		return 0.013152386
	}
	return -0.047765612999999998
}
