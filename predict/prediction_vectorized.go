package predict

import (
	"encoding/csv"
	"go/linear/gradient/hypothesis"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func PredictVectorized(predictionFile string, theta mat.Matrix, M mat.Matrix, S mat.Matrix) (mat.Matrix, [][]string, error) {
	f, err := os.Open(predictionFile)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	predictData := mat.NewDense(len(lines), len(lines[0])+1, nil)
	for i, line := range lines {
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, err
			}

			if err != nil {
				return nil, nil, err
			}
			if j == 0 {
				predictData.Set(i, 0, 1)
			}
			predictData.Set(i, j+1, f)

		}
	}

	// Normalize input
	predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v - M.At(0, j-1)
	}, predictData)

	predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v / S.At(0, j-1)
	}, predictData)

	// Calc h(x)
	result, err := hypothesis.ComputeHypothesisVectorized(theta, predictData)
	if err != nil {
		return nil, nil, err
	}
	return result, lines, err
}
