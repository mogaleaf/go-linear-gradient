package slice

import (
	"encoding/csv"
	"fmt"
	"go/linear/gradient/hypothesis"
	"go/linear/gradient/predict"
	"os"
	"strconv"
)

type predictSlice struct {
	theta []float64
	M     []float64
	S     []float64
}

func NewSlicePredict(theta []float64, M []float64, S []float64) predict.Predict {
	return &predictSlice{
		theta: theta,
		M:     M,
		S:     S,
	}
}

func (p *predictSlice) Predict(predictionFile string, resultFile string) error {
	f, err := os.Open(predictionFile)
	if err != nil {
		return err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return err
	}

	predictData := make([][]float64, len(lines))

	for i, line := range lines {
		predictData[i] = make([]float64, len(line))
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return err
			}
			predictData[i][j] = f
		}
	}

	//First normalize
	for i := 0; i < len(predictData); i++ {
		for j := 0; j < len(predictData[i]); j++ {
			predictData[i][j] = (predictData[i][j] - p.M[j]) / p.S[j]
		}

	}

	//Then give result
	result := make([]float64, len(predictData))
	for i := 0; i < len(predictData); i++ {
		result[i] = hypothesis.ComputeHypothesis(predictData[i], p.theta)
	}

	err = writeData(result, resultFile)

	return err
}

func writeData(result []float64, resultFile string) error {
	f, err := os.Create(resultFile)
	if err != nil {
		return err
	}
	defer f.Close()
	writer := csv.NewWriter(f)
	for i := 0; i < len(result); i++ {
		writer.Write([]string{fmt.Sprintf("%0.10f", result[i])})
		println(fmt.Sprintf("prediction %0.10f", result[i]))

	}
	writer.Flush()
	return nil
}
