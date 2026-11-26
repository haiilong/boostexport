package xgb

const (
	baseScore = 6.6373749999999996
)

// Predict returns the regression score for feature vector f.
func Predict(f []float64) float64 {
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
	s += tree40(f)
	s += tree41(f)
	s += tree42(f)
	s += tree43(f)
	s += tree44(f)
	s += tree45(f)
	s += tree46(f)
	s += tree47(f)
	s += tree48(f)
	s += tree49(f)
	return s
}

func tree0(f []float64) float64 {
	if float32(f[3]) < -0.249036044 {
		if float32(f[7]) < -0.246062487 {
			if float32(f[3]) < -0.542673826 {
				return -20.458760000000002
			}
			return -7.8773980000000003
		}
		if float32(f[7]) < 0.95312506 {
			return -6.4086350000000003
		}
		return 2.6472000000000002
	}
	if float32(f[7]) < 0.213196635 {
		if float32(f[3]) < 0.827183247 {
			return -1.3389791
		}
		return 9.2581624999999992
	}
	if float32(f[3]) < 0.0155790476 {
		return 4.7365145999999996
	}
	return 16.266666000000001
}

func tree1(f []float64) float64 {
	if float32(f[3]) < -0.411876976 {
		if float32(f[7]) < 0.22468473 {
			if float32(f[3]) < -1.00252938 {
				return -18.900151999999999
			}
			return -10.551923
		}
		if float32(f[1]) < 1.45114362 {
			return -3.6085319999999999
		}
		return 7.7627550000000003
	}
	if float32(f[7]) < 0.351448208 {
		if float32(f[3]) < 0.827183247 {
			return -1.5754477
		}
		return 8.5178609999999999
	}
	if float32(f[3]) < 0.0155790476 {
		return 3.7123737000000001
	}
	return 15.293262
}

func tree2(f []float64) float64 {
	if float32(f[3]) < -0.249036044 {
		if float32(f[7]) < -0.246062487 {
			if float32(f[3]) < -1.00252938 {
				return -19.140374999999999
			}
			return -9.4632729999999992
		}
		if float32(f[7]) < 0.95312506 {
			return -5.2068477
		}
		return 2.4139716999999998
	}
	if float32(f[7]) < 0.181866258 {
		if float32(f[1]) < 0.132969677 {
			return -2.1251698000000001
		}
		return 7.3247422999999996
	}
	if float32(f[6]) < 0.384065449 {
		return 7.9400940000000002
	}
	return 17.030446999999999
}

func tree3(f []float64) float64 {
	if float32(f[3]) < -0.411876976 {
		if float32(f[7]) < 0.949554145 {
			if float32(f[7]) < -0.372206777 {
				return -14.767438
			}
			return -7.5493059999999996
		}
		if float32(f[1]) < 1.45114362 {
			return -0.61809707000000003
		}
		return 6.9146953
	}
	if float32(f[7]) < 0.0824397504 {
		if float32(f[3]) < 0.827183247 {
			return -2.4938210999999999
		}
		return 6.6964354999999998
	}
	if float32(f[6]) < 0.586693823 {
		return 6.0415780000000003
	}
	return 15.118281
}

func tree4(f []float64) float64 {
	if float32(f[3]) < -0.0807165802 {
		if float32(f[7]) < -0.246062487 {
			if float32(f[3]) < -1.00252938 {
				return -16.03397
			}
			return -7.5522780000000003
		}
		if float32(f[7]) < 0.95312506 {
			return -3.8539319999999999
		}
		return 2.0794969999999999
	}
	if float32(f[7]) < 0.559790432 {
		if float32(f[1]) < 0.323167711 {
			return -0.90912782999999997
		}
		return 8.946593
	}
	if float32(f[1]) < -0.624818563 {
		return 5.799785
	}
	return 14.060895
}

func tree5(f []float64) float64 {
	if float32(f[3]) < -0.511215687 {
		if float32(f[7]) < 0.22468473 {
			if float32(f[1]) < 0.632407725 {
				return -13.295481000000001
			}
			return -4.2073700000000001
		}
		if float32(f[1]) < 0.632781863 {
			return -3.9141604999999999
		}
		return 2.8723814000000001
	}
	if float32(f[3]) < 0.883659959 {
		if float32(f[7]) < -0.00360253919 {
			return -2.6002057000000001
		}
		return 4.5269674999999996
	}
	if float32(f[7]) < -0.482743531 {
		return 1.5706226000000001
	}
	return 12.019667
}

func tree6(f []float64) float64 {
	if float32(f[3]) < -0.0807165802 {
		if float32(f[7]) < -0.246062487 {
			if float32(f[3]) < -1.00252938 {
				return -13.433928999999999
			}
			return -6.2059063999999999
		}
		if float32(f[1]) < -0.00607091142 {
			return -3.9606892999999999
		}
		return 1.2987542000000001
	}
	if float32(f[7]) < -0.446183443 {
		if float32(f[1]) < 0.279968619 {
			return -3.5288092999999998
		}
		return 4.2348876000000004
	}
	if float32(f[3]) < 1.84670734 {
		return 5.6331420000000003
	}
	return 16.731836000000001
}

func tree7(f []float64) float64 {
	if float32(f[3]) < -0.511215687 {
		if float32(f[7]) < 0.22468473 {
			if float32(f[6]) < -0.923233271 {
				return -16.199919000000001
			}
			return -7.3143343999999999
		}
		if float32(f[1]) < 1.45114362 {
			return -2.6412236999999998
		}
		return 5.0018339999999997
	}
	if float32(f[3]) < 0.883659959 {
		if float32(f[7]) < 1.19504666 {
			return -0.67721956999999999
		}
		return 7.8194869999999996
	}
	if float32(f[1]) < -0.645964205 {
		return 1.4643204000000001
	}
	return 10.363575000000001
}

func tree8(f []float64) float64 {
	if float32(f[3]) < -0.511215687 {
		if float32(f[7]) < 0.949554145 {
			if float32(f[6]) < -0.923233271 {
				return -14.265953
			}
			return -5.8627479999999998
		}
		if float32(f[3]) < -1.15836465 {
			return -2.0368822
		}
		return 2.9134726999999998
	}
	if float32(f[7]) < -0.649277568 {
		if float32(f[3]) < 1.57398677 {
			return -4.5886940000000003
		}
		return 4.1360964999999998
	}
	if float32(f[3]) < 0.633919001 {
		return 1.6872682999999999
	}
	return 8.0487929999999999
}

func tree9(f []float64) float64 {
	if float32(f[3]) < 0.0210038424 {
		if float32(f[7]) < -0.44550252 {
			if float32(f[3]) < -0.542673826 {
				return -9.8062039999999993
			}
			return -3.8445680000000002
		}
		if float32(f[1]) < -0.00607091142 {
			return -3.5003033000000001
		}
		return 1.204882
	}
	if float32(f[7]) < 0.559790432 {
		if float32(f[1]) < 0.279968619 {
			return -1.3630947
		}
		return 6.1284156000000003
	}
	if float32(f[6]) < -0.926930487 {
		return 2.9455173000000001
	}
	return 10.870105000000001
}

func tree10(f []float64) float64 {
	if float32(f[3]) < -0.767347574 {
		if float32(f[1]) < 0.632407725 {
			if float32(f[7]) < 0.126379579 {
				return -9.6237879999999993
			}
			return -3.3442509999999999
		}
		if float32(f[6]) < 0.778361082 {
			return -1.7195777999999999
		}
		return 3.7582479000000002
	}
	if float32(f[3]) < 1.84670734 {
		if float32(f[7]) < 1.72754312 {
			return 0.43007255
		}
		return 11.742513000000001
	}
	if float32(f[1]) < -0.966976166 {
		return 4.5009645999999996
	}
	return 14.08497
}

func tree11(f []float64) float64 {
	if float32(f[3]) < 0.0210038424 {
		if float32(f[7]) < -0.44550252 {
			if float32(f[6]) < -0.790474474 {
				return -11.362334000000001
			}
			return -5.4462875999999998
		}
		if float32(f[1]) < -0.00607091142 {
			return -2.9995449999999999
		}
		return 1.1232802
	}
	if float32(f[6]) < 0.519346535 {
		if float32(f[7]) < -0.446183443 {
			return -2.9417502999999998
		}
		return 2.8021848
	}
	if float32(f[7]) < -0.192360967 {
		return 3.0473096000000002
	}
	return 9.7616610000000001
}

func tree12(f []float64) float64 {
	if float32(f[3]) < -0.714351416 {
		if float32(f[1]) < 0.632407725 {
			if float32(f[7]) < -0.550305128 {
				return -9.3183509999999998
			}
			return -4.3117904999999999
		}
		if float32(f[7]) < 1.30547881 {
			return -1.2866466999999999
		}
		return 3.7608635000000001
	}
	if float32(f[1]) < 0.793488741 {
		if float32(f[7]) < 0.559790432 {
			return -1.2494240999999999
		}
		return 3.9889190000000001
	}
	if float32(f[7]) < -0.595661283 {
		return 0.34246286999999997
	}
	return 7.7238344999999997
}

func tree13(f []float64) float64 {
	if float32(f[3]) < 0.633919001 {
		if float32(f[7]) < -0.643518209 {
			if float32(f[6]) < -0.446514964 {
				return -9.8179780000000001
			}
			return -4.2995979999999996
		}
		if float32(f[7]) < 1.5231241 {
			return -1.0068402000000001
		}
		return 5.4673686000000004
	}
	if float32(f[7]) < -0.576510429 {
		if float32(f[0]) < 0.444602668 {
			return -2.5089703000000001
		}
		return 3.60371
	}
	if float32(f[1]) < -0.637387156 {
		return 1.4424051
	}
	return 8.254073
}

func tree14(f []float64) float64 {
	if float32(f[3]) < -0.767347574 {
		if float32(f[1]) < 0.632407725 {
			if float32(f[7]) < 0.126379579 {
				return -7.1425247000000001
			}
			return -2.5258927
		}
		if float32(f[6]) < 0.778361082 {
			return -1.4278223999999999
		}
		return 3.2016550000000001
	}
	if float32(f[1]) < 0.869605899 {
		if float32(f[7]) < 0.621571898 {
			return -1.1205400000000001
		}
		return 3.5601978000000001
	}
	if float32(f[7]) < -0.602212071 {
		return 0.18158087000000001
	}
	return 6.6737323000000002
}

func tree15(f []float64) float64 {
	if float32(f[3]) < 0.633919001 {
		if float32(f[6]) < -0.790474474 {
			if float32(f[7]) < 0.536509871 {
				return -7.6903709999999998
			}
			return -0.60174625999999998
		}
		if float32(f[7]) < 1.50235701 {
			return -0.89110243
		}
		return 5.8804536000000001
	}
	if float32(f[3]) < 1.84670734 {
		if float32(f[7]) < -0.808493614 {
			return -2.3478444000000001
		}
		return 3.7390213000000001
	}
	if float32(f[1]) < -1.37931919 {
		return 2.2626575999999998
	}
	return 10.065778999999999
}

func tree16(f []float64) float64 {
	if float32(f[3]) < -1.15836465 {
		if float32(f[7]) < 0.126379579 {
			if float32(f[6]) < -0.10444922 {
				return -9.0899370000000008
			}
			return -4.0936775000000001
		}
		if float32(f[3]) < -1.59470308 {
			return -3.5047633999999999
		}
		return -0.96142309999999997
	}
	if float32(f[7]) < -0.981508672 {
		if float32(f[6]) < 0.388578594 {
			return -6.0318623000000002
		}
		return -2.1103475
	}
	if float32(f[1]) < -0.581680894 {
		return -1.4363767000000001
	}
	return 2.9024972999999998
}

func tree17(f []float64) float64 {
	if float32(f[3]) < 0.633919001 {
		if float32(f[7]) < -0.643518209 {
			if float32(f[1]) < 0.538756013 {
				return -5.7843165000000001
			}
			return -0.48382797999999999
		}
		if float32(f[6]) < -0.792520761 {
			return -3.9706885999999999
		}
		return 0.64509976000000002
	}
	if float32(f[3]) < 1.84670734 {
		if float32(f[7]) < -0.808493614 {
			return -1.8840302
		}
		return 3.2347785999999998
	}
	if float32(f[1]) < -1.37931919 {
		return 2.2213438000000001
	}
	return 8.9852910000000001
}

func tree18(f []float64) float64 {
	if float32(f[3]) < 0.184680313 {
		if float32(f[7]) < -0.246062487 {
			if float32(f[6]) < -0.828994989 {
				return -8.4862280000000005
			}
			return -2.7420404
		}
		if float32(f[7]) < 2.14394403 {
			return -0.64907689999999996
		}
		return 6.0819200000000002
	}
	if float32(f[7]) < 0.565509617 {
		if float32(f[1]) < 0.0754336417 {
			return -1.1506007
		}
		return 3.2893894000000001
	}
	if float32(f[1]) < -0.624818563 {
		return 1.8768636999999999
	}
	return 7.034065
}

func tree19(f []float64) float64 {
	if float32(f[3]) < -1.15836465 {
		if float32(f[7]) < 0.126379579 {
			if float32(f[6]) < -0.923233271 {
				return -8.064368
			}
			return -3.6177975999999998
		}
		if float32(f[9]) < 0.674819469 {
			return -0.86309990000000003
		}
		return -3.2663864999999999
	}
	if float32(f[7]) < -0.981508672 {
		if float32(f[7]) < -1.51936996 {
			return -5.8691835000000001
		}
		return -2.1945958000000001
	}
	if float32(f[6]) < -0.0153098488 {
		return -0.53818880000000002
	}
	return 2.9274749999999998
}

func tree20(f []float64) float64 {
	if float32(f[3]) < 1.39568377 {
		if float32(f[6]) < -0.790474474 {
			if float32(f[7]) < 0.536509871 {
				return -5.4308579999999997
			}
			return -0.10279070999999999
		}
		if float32(f[7]) < 1.30547881 {
			return -0.32183816999999998
		}
		return 4.7318660000000001
	}
	if float32(f[7]) < -0.434496224 {
		if float32(f[0]) < -1.19787788 {
			return -1.4491388999999999
		}
		return 2.9394474000000002
	}
	if float32(f[1]) < -1.37190115 {
		return 2.6518671999999999
	}
	return 7.6090600000000004
}

func tree21(f []float64) float64 {
	if float32(f[3]) < 0.883659959 {
		if float32(f[7]) < 1.30547881 {
			if float32(f[6]) < -0.446514964 {
				return -3.8590412000000001
			}
			return -0.47768359999999999
		}
		if float32(f[1]) < -0.268888682 {
			return -0.1311541
		}
		return 4.7493166999999996
	}
	if float32(f[6]) < -1.22954965 {
		if float32(f[1]) < -1.37931919 {
			return 1.9107491000000001
		}
		return -3.1771069000000001
	}
	if float32(f[3]) < 1.3686316 {
		return 2.2985712999999999
	}
	return 5.9852540000000003
}

func tree22(f []float64) float64 {
	if float32(f[1]) < 0.869605899 {
		if float32(f[3]) < -0.714351416 {
			if float32(f[3]) < -1.57022476 {
				return -5.1819389999999999
			}
			return -2.1359873
		}
		if float32(f[6]) < 1.19663143 {
			return -0.35242972
		}
		return 4.1523320000000004
	}
	if float32(f[3]) < -0.919384599 {
		if float32(f[6]) < 0.361395597 {
			return -1.0248117000000001
		}
		return 1.2492536000000001
	}
	if float32(f[7]) < -0.234153375 {
		return 1.4886192
	}
	return 4.7203470000000003
}

func tree23(f []float64) float64 {
	if float32(f[3]) < 0.248220593 {
		if float32(f[7]) < 1.03753996 {
			if float32(f[6]) < -0.828994989 {
				return -5.3618759999999996
			}
			return -1.1777693
		}
		if float32(f[1]) < -0.268888682 {
			return -0.43038678000000002
		}
		return 2.9713042000000001
	}
	if float32(f[7]) < -1.2221278 {
		return -4.987107
	}
	if float32(f[6]) < -1.22954965 {
		return -1.9160497999999999
	}
	return 2.6565120000000002
}

func tree24(f []float64) float64 {
	if float32(f[3]) < -0.891192198 {
		if float32(f[3]) < -1.57022476 {
			if float32(f[8]) < 0.267392308 {
				return -4.9638179999999998
			}
			return -1.5455798000000001
		}
		if float32(f[1]) < -1.4480139 {
			return -5.214785
		}
		return -0.69170690000000001
	}
	if float32(f[1]) < -0.581680894 {
		if float32(f[3]) < 1.57398677 {
			return -1.7387087000000001
		}
		return 2.7463538999999999
	}
	if float32(f[7]) < -0.180479914 {
		return -0.28814504000000002
	}
	return 2.7367404
}

func tree25(f []float64) float64 {
	if float32(f[1]) < 0.869605899 {
		if float32(f[7]) < -0.112328053 {
			if float32(f[3]) < 0.622849941 {
				return -2.7182727
			}
			return 0.094418370000000001
		}
		if float32(f[6]) < 1.44127333 {
			return 0.034459642999999998
		}
		return 6.0241674999999999
	}
	if float32(f[3]) < 1.08004808 {
		if float32(f[7]) < 1.03753996 {
			return 1.1011947
		}
		return 3.607472
	}
	if float32(f[9]) < -0.897254348 {
		return 6.2716865999999998
	}
	return 1.5975546
}

func tree26(f []float64) float64 {
	if float32(f[3]) < 1.63911676 {
		if float32(f[6]) < -0.0716012567 {
			if float32(f[7]) < 0.812525809 {
				return -2.6295497000000001
			}
			return 0.30877349999999998
		}
		if float32(f[7]) < -0.981508672 {
			return -2.4717232999999998
		}
		return 1.2538324999999999
	}
	if float32(f[3]) < 1.84670734 {
		return 1.7359422
	}
	if float32(f[2]) < -0.999302208 {
		return 1.0556258999999999
	}
	return 5.6692879999999999
}

func tree27(f []float64) float64 {
	if float32(f[3]) < 0.184680313 {
		if float32(f[7]) < -0.753964603 {
			if float32(f[1]) < 0.538756013 {
				return -4.4514427000000003
			}
			return 0.030027390000000001
		}
		if float32(f[7]) < 2.14394403 {
			return -0.65978270000000006
		}
		return 4.1496462999999997
	}
	if float32(f[7]) < 0.565509617 {
		if float32(f[7]) < -1.2221278 {
			return -4.1722045000000003
		}
		return 0.89015955000000002
	}
	if float32(f[6]) < -0.926930487 {
		return 1.2151068
	}
	return 4.4403853
}

func tree28(f []float64) float64 {
	if float32(f[3]) < -1.22084367 {
		if float32(f[0]) < 0.130740583 {
			if float32(f[3]) < -1.6163106 {
				return -3.2156456000000002
			}
			return -0.054210313000000003
		}
		if float32(f[4]) < 0.779660523 {
			return -4.4209733
		}
		return -0.13145982000000001
	}
	if float32(f[1]) < -0.581680894 {
		if float32(f[3]) < 1.57398677 {
			return -1.4405683
		}
		return 2.1345453000000001
	}
	if float32(f[7]) < -0.981508672 {
		return -2.0918739999999998
	}
	return 1.5710871
}

func tree29(f []float64) float64 {
	if float32(f[3]) < 1.39568377 {
		if float32(f[6]) < -0.0716012567 {
			if float32(f[7]) < 0.812525809 {
				return -2.4378476
			}
			return 0.15553191
		}
		if float32(f[7]) < 1.30434024 {
			return 0.19745778999999999
		}
		return 3.997004
	}
	if float32(f[7]) < -0.434496224 {
		if float32(f[2]) < -0.773789227 {
			return -0.40216294000000002
		}
		return 1.9674993999999999
	}
	if float32(f[2]) < -0.999302208 {
		return 0.87978213999999999
	}
	return 4.5215180000000004
}

func tree30(f []float64) float64 {
	if float32(f[3]) < -0.891192198 {
		if float32(f[3]) < -1.76076281 {
			if float32(f[1]) < 0.857659638 {
				return -4.2148724
			}
			return -1.1174105000000001
		}
		if float32(f[1]) < -1.4480139 {
			return -3.9925457999999998
		}
		return -0.82801020000000003
	}
	if float32(f[1]) < -0.581680894 {
		if float32(f[7]) < 1.02257049 {
			return -1.5176609999999999
		}
		return 0.87792455999999996
	}
	if float32(f[7]) < -0.981508672 {
		return -1.6024092000000001
	}
	return 1.5043234999999999
}

func tree31(f []float64) float64 {
	if float32(f[6]) < -0.790474474 {
		if float32(f[7]) < 0.536509871 {
			if float32(f[4]) < 0.919154167 {
				return -3.2499359000000001
			}
			return 0.50637810000000005
		}
		if float32(f[3]) < 0.538910031 {
			return -0.41853052000000002
		}
		return 1.4865379000000001
	}
	if float32(f[3]) < 1.3686316 {
		if float32(f[7]) < 1.30547881 {
			return -0.19549491999999999
		}
		return 2.6583985999999999
	}
	if float32(f[7]) < -0.434496224 {
		return 1.0963590999999999
	}
	return 4.4623160000000004
}

func tree32(f []float64) float64 {
	if float32(f[1]) < -1.39856756 {
		if float32(f[7]) < 0.126379579 {
			return -4.4627213000000001
		}
		if float32(f[0]) < -0.0555476993 {
			return -2.8559315000000001
		}
		return -0.33021050000000002
	}
	if float32(f[3]) < -1.57022476 {
		if float32(f[0]) < -2.15289068 {
			return 0.030210113
		}
		return -3.2332207999999998
	}
	if float32(f[7]) < -1.51936996 {
		return -3.5312665000000001
	}
	return 0.68408316000000002
}

func tree33(f []float64) float64 {
	if float32(f[6]) < 0.570890486 {
		if float32(f[3]) < 1.63911676 {
			if float32(f[1]) < -0.024354592 {
				return -1.5077308
			}
			return 0.11281053000000001
		}
		if float32(f[0]) < 1.18839324 {
			return 3.9129969999999998
		}
		return 0.99059724999999998
	}
	if float32(f[3]) < 0.0592184328 {
		if float32(f[9]) < -0.0595253557 {
			return -0.96263533999999995
		}
		return 1.418201
	}
	if float32(f[2]) < 0.607896507 {
		return 2.9203220000000001
	}
	return 0.63003399999999998
}

func tree34(f []float64) float64 {
	if float32(f[3]) < 0.633919001 {
		if float32(f[6]) < 0.133540899 {
			if float32(f[8]) < -0.88749218 {
				return -3.0453174000000001
			}
			return -0.71270420000000001
		}
		if float32(f[7]) < -0.989628136 {
			return -1.9684025000000001
		}
		return 0.86930704000000003
	}
	if float32(f[7]) < 0.565509617 {
		if float32(f[6]) < -1.22954965 {
			return -1.7527071000000001
		}
		return 0.97461193999999995
	}
	if float32(f[1]) < -1.09231281 {
		return 0.60363690000000003
	}
	return 3.8698489999999999
}

func tree35(f []float64) float64 {
	if float32(f[1]) < 0.869605899 {
		if float32(f[6]) < 1.53075087 {
			if float32(f[7]) < -1.51936996 {
				return -3.7902149999999999
			}
			return -0.41827365999999999
		}
		if float32(f[1]) < -0.704343677 {
			return -2.1043541000000001
		}
		return 3.5669749999999998
	}
	if float32(f[3]) < -1.07008481 {
		if float32(f[1]) < 1.40279436 {
			return 0.14141819999999999
		}
		return -1.0161469000000001
	}
	if float32(f[7]) < 1.17944014 {
		return 1.2772654000000001
	}
	return 3.6943853
}

func tree36(f []float64) float64 {
	if float32(f[1]) < -1.39856756 {
		if float32(f[7]) < 0.126379579 {
			return -3.9085185999999998
		}
		if float32(f[0]) < -0.0555476993 {
			return -2.4250683999999998
		}
		return -0.21152750000000001
	}
	if float32(f[3]) < -1.57022476 {
		if float32(f[3]) < -1.95208776 {
			return -3.5824668000000002
		}
		return -1.6540197000000001
	}
	if float32(f[7]) < 1.5231241 {
		return 0.17882609999999999
	}
	return 2.5902107000000001
}

func tree37(f []float64) float64 {
	if float32(f[6]) < -0.790474474 {
		if float32(f[7]) < 0.536509871 {
			if float32(f[4]) < 0.919154167 {
				return -2.5405376
			}
			return 0.30046842000000001
		}
		if float32(f[5]) < -0.250046521 {
			return -1.0510113000000001
		}
		return 0.56432055999999997
	}
	if float32(f[7]) < -0.981508672 {
		if float32(f[8]) < -0.220964178 {
			return -3.5517788000000001
		}
		return -0.62973409999999996
	}
	if float32(f[3]) < 0.11092259 {
		return -0.089581599999999997
	}
	return 1.5859722000000001
}

func tree38(f []float64) float64 {
	if float32(f[1]) < -1.39856756 {
		if float32(f[6]) < -0.551858127 {
			return -3.8360504999999998
		}
		if float32(f[2]) < -0.998385429 {
			return 0.023896283000000001
		}
		return -1.8526658
	}
	if float32(f[7]) < -1.51936996 {
		if float32(f[1]) < 0.538756013 {
			return -3.3361293999999999
		}
		return 0.93683857000000004
	}
	if float32(f[3]) < -1.59470308 {
		return -2.2237024000000001
	}
	return 0.49700480000000002
}

func tree39(f []float64) float64 {
	if float32(f[6]) < -0.790474474 {
		if float32(f[7]) < 0.536509871 {
			if float32(f[4]) < -0.518270195 {
				return -3.3513226999999999
			}
			return -1.2454706
		}
		if float32(f[5]) < -0.250046521 {
			return -1.0094607
		}
		return 0.46998056999999999
	}
	if float32(f[3]) < 1.3686316 {
		if float32(f[7]) < 1.30547881 {
			return -0.16017948000000001
		}
		return 1.9518979999999999
	}
	if float32(f[2]) < -0.999302208 {
		return 0.067216999999999999
	}
	return 3.1327061999999999
}

func tree40(f []float64) float64 {
	if float32(f[6]) < 1.19663143 {
		if float32(f[1]) < 0.632781863 {
			if float32(f[7]) < 1.02257049 {
				return -0.92185514999999996
			}
			return 0.73151535000000001
		}
		if float32(f[3]) < -1.48489797 {
			return -1.2026918
		}
		return 1.0344941999999999
	}
	if float32(f[1]) < -0.704343677 {
		if float32(f[0]) < 0.55148536 {
			return 0.054830549999999999
		}
		return -1.5640305000000001
	}
	if float32(f[7]) < -0.170184627 {
		return 1.1808730000000001
	}
	return 3.2914797999999998
}

func tree41(f []float64) float64 {
	if float32(f[3]) < 0.428307086 {
		if float32(f[6]) < 0.0242194571 {
			if float32(f[9]) < -0.0555846691 {
				return -2.0321207000000001
			}
			return -0.43645200000000001
		}
		if float32(f[7]) < 0.0818293616 {
			return -0.46277829999999998
		}
		return 1.0177251
	}
	if float32(f[7]) < -1.2221278 {
		return -2.282311
	}
	if float32(f[3]) < 2.00609279 {
		return 0.87121150000000003
	}
	return 3.0619543
}

func tree42(f []float64) float64 {
	if float32(f[1]) < -1.39856756 {
		if float32(f[6]) < -0.551858127 {
			return -3.1625960000000002
		}
		if float32(f[2]) < -0.998385429 {
			return -0.0010818481999999999
		}
		return -1.5785395
	}
	if float32(f[3]) < -1.57022476 {
		if float32(f[3]) < -1.95208776 {
			return -2.8163914999999999
		}
		return -1.0896451
	}
	if float32(f[6]) < 1.10552597 {
		return 0.099125400000000002
	}
	return 1.915835
}

func tree43(f []float64) float64 {
	if float32(f[6]) < -0.790474474 {
		if float32(f[3]) < 1.39568377 {
			if float32(f[7]) < 1.5231241 {
				return -1.6479988000000001
			}
			return 0.33235836000000002
		}
		if float32(f[0]) < -1.19787788 {
			return -0.30188579999999998
		}
		return 1.1784912000000001
	}
	if float32(f[7]) < -0.643518209 {
		if float32(f[4]) < -0.525755048 {
			return -2.2268772000000001
		}
		return -0.28190683999999999
	}
	if float32(f[3]) < 0.0210038424 {
		return -0.048299950000000001
	}
	return 1.2259378000000001
}

func tree44(f []float64) float64 {
	if float32(f[7]) < 1.54210997 {
		if float32(f[6]) < -0.307778239 {
			if float32(f[4]) < -0.518270195 {
				return -1.9614042
			}
			return -0.34254086
		}
		if float32(f[3]) < 1.24774206 {
			return -0.0078901014999999998
		}
		return 2.0846309999999999
	}
	if float32(f[3]) < -0.241257653 {
		if float32(f[7]) < 2.14394403 {
			return 0.020350073999999999
		}
		return 1.5757884
	}
	if float32(f[2]) < -0.676392317 {
		return 0.57502339999999996
	}
	return 3.1560917000000002
}

func tree45(f []float64) float64 {
	if float32(f[1]) < -1.39856756 {
		if float32(f[7]) < 1.29699457 {
			if float32(f[4]) < 0.894924402 {
				return -2.223001
			}
			return -0.029540253999999998
		}
		if float32(f[0]) < 0.55148536 {
			return 0.0040664673000000004
		}
		return -0.24070016
	}
	if float32(f[7]) < -1.51936996 {
		if float32(f[1]) < 0.538756013 {
			return -2.60541
		}
		return 0.78079909999999997
	}
	if float32(f[3]) < -1.59470308 {
		return -1.6322277999999999
	}
	return 0.37043350000000003
}

func tree46(f []float64) float64 {
	if float32(f[7]) < 1.03099954 {
		if float32(f[4]) < -0.077837199 {
			if float32(f[1]) < -0.238948047 {
				return -1.6933883000000001
			}
			return -0.32809767000000001
		}
		if float32(f[3]) < 1.35624003 {
			return -0.011204463
		}
		return 1.9643933
	}
	if float32(f[1]) < -1.07089245 {
		if float32(f[1]) < -2.06744218 {
			return 0.0038631440000000002
		}
		return -0.62325494999999997
	}
	if float32(f[3]) < -1.03990591 {
		return -0.21360496000000001
	}
	return 1.7705811
}

func tree47(f []float64) float64 {
	if float32(f[1]) < 0.632781863 {
		if float32(f[6]) < 1.53075087 {
			if float32(f[3]) < 0.412780941 {
				return -0.75497216
			}
			return 0.27635739999999998
		}
		if float32(f[1]) < -0.704343677 {
			return -1.0815292999999999
		}
		return 2.7029977000000001
	}
	if float32(f[3]) < -1.48489797 {
		return -0.89097400000000004
	}
	if float32(f[7]) < 0.996266842 {
		return 0.54838399999999998
	}
	return 1.8140272
}

func tree48(f []float64) float64 {
	if float32(f[3]) < 2.00609279 {
		if float32(f[6]) < 0.308050543 {
			if float32(f[7]) < 1.03753996 {
				return -0.70714069999999996
			}
			return 0.51629369999999997
		}
		if float32(f[7]) < -0.315269232 {
			return -0.37952572000000001
		}
		return 1.0107429999999999
	}
	if float32(f[1]) < -1.37931919 {
		return 0.56656499999999999
	}
	return 2.7613704000000001
}

func tree49(f []float64) float64 {
	if float32(f[6]) < -0.828994989 {
		if float32(f[3]) < 1.39568377 {
			if float32(f[9]) < -0.0555846691 {
				return -1.7311658000000001
			}
			return -0.48250707999999998
		}
		if float32(f[0]) < -1.19787788 {
			return -0.36486669999999999
		}
		return 0.96061209999999997
	}
	if float32(f[7]) < -1.2221278 {
		if float32(f[9]) < 1.61558282 {
			return -1.7007555000000001
		}
		return 0.73387639999999998
	}
	if float32(f[3]) < -1.33123291 {
		return -0.95835000000000004
	}
	return 0.45965716000000001
}
