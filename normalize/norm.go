package normalize

import (
	"errors"
	"math"
)

func Normalize(data [][]float64) ([][]float64, []float64, []float64, error) {
	r := len(data)
	c := len(data[0])

	minMax := make([]float64, c)
	means := make([]float64, c)
	dataNorm := make([][]float64, r)
	for i := 0; i < r; i++ {
		dataNorm[i] = make([]float64, c)
	}
	for j := 0; j < c; j++ {
		var max float64
		min := math.MaxFloat64
		sum := 0.0
		for i := 0; i < r; i++ {
			if max < data[i][j] {
				max = data[i][j]
			}
			if min > data[i][j] {
				min = data[i][j]
			}
			sum += data[i][j]
		}
		means[j] = sum / float64(r)
		minMax[j] = max - min
		if minMax[j] == 0 {
			return nil, nil, nil, errors.New("Min max should not be 0")
		}
		for i := 0; i < r; i++ {
			dataNorm[i][j] = (data[i][j] - means[j]) / minMax[j]
		}
	}

	return dataNorm, means, minMax, nil
}
