package slice

import (
	"encoding/csv"
	"go/linear/gradient/hypothesis"
	"go/linear/gradient/predict"
	"os"
	"strconv"
)

type predictSlice struct {
	theta       []float64
	M           []float64
	S           []float64
	predictData [][]float64
}

func NewSlicePredict(predictionFile string, theta []float64, M []float64, S []float64) (predict.Predict, error) {
	initPredictData, err := initPredictData(predictionFile)
	if err != nil {
		return nil, err
	}
	return &predictSlice{
		theta:       theta,
		M:           M,
		S:           S,
		predictData: initPredictData,
	}, nil
}

func (p *predictSlice) PredictLength() int {
	return len(p.predictData)
}

func (p *predictSlice) Predict(resultData chan float64) {

	//First normalize
	for i := 0; i < len(p.predictData); i++ {
		for j := 0; j < len(p.predictData[i]); j++ {
			p.predictData[i][j] = (p.predictData[i][j] - p.M[j]) / p.S[j]
		}

	}

	//Then give result
	for i := 0; i < len(p.predictData); i++ {
		resultData <- hypothesis.ComputeHypothesis(p.predictData[i], p.theta)
	}
	close(resultData)
}

func initPredictData(predictionFile string) ([][]float64, error) {
	f, err := os.Open(predictionFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}

	predictData := make([][]float64, len(lines))

	for i, line := range lines {
		predictData[i] = make([]float64, len(line))
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, err
			}
			predictData[i][j] = f
		}
	}
	return predictData, nil
}
