package predict

import (
	"encoding/csv"
	"go/linear/gradient/hypothesis"
	"os"
	"strconv"
)

func Predict(predictionFile string, theta []float64, M []float64, S []float64) ([]float64, [][]string, error) {
	f, err := os.Open(predictionFile)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	predictData := make([][]float64, len(lines))

	for i, line := range lines {
		predictData[i] = make([]float64, len(line))
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, err
			}
			predictData[i][j] = f
		}
	}
	for i := 0; i < len(predictData); i++ {
		for j := 0; j < len(predictData[i]); j++ {
			predictData[i][j] = (predictData[i][j] - M[j]) / S[j]
		}

	}

	result := make([]float64, len(predictData))
	for i := 0; i < len(predictData); i++ {
		result[i] = hypothesis.ComputeHypothesis(predictData[i], theta)
	}

	return result, lines, err
}
