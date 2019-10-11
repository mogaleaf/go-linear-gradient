package vectorized

import (
	"encoding/csv"
	"go/linear/gradient/hypothesis"
	"go/linear/gradient/predict"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type predictVectorized struct {
	theta       mat.Matrix
	M           mat.Matrix
	S           mat.Matrix
	predictData *mat.Dense
}

func NewPredictVectorized(predictionfile string, theta mat.Matrix, M mat.Matrix, S mat.Matrix) (predict.Predict, error) {
	matrix, err := initPredictData(predictionfile)
	if err != nil {
		return nil, err
	}
	return &predictVectorized{
		S:           S,
		M:           M,
		theta:       theta,
		predictData: matrix,
	}, nil
}

func (p *predictVectorized) PredictLength() int {
	r, _ := p.predictData.Dims()
	return r
}

func (p *predictVectorized) Predict(resultData chan float64) {

	// Normalize input
	p.predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v - p.M.At(0, j-1)
	}, p.predictData)

	p.predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v / p.S.At(0, j-1)
	}, p.predictData)

	// Calc h(x)
	result := hypothesis.ComputeHypothesisVectorized(p.theta, p.predictData)
	r, _ := result.Dims()
	for i := 0; i < r; i++ {
		resultData <- result.At(i, 0)
	}
	close(resultData)

}

func initPredictData(predictionFile string) (*mat.Dense, error) {
	f, err := os.Open(predictionFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}

	predictData := mat.NewDense(len(lines), len(lines[0])+1, nil)
	for i, line := range lines {
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, err
			}

			if err != nil {
				return nil, err
			}
			if j == 0 {
				predictData.Set(i, 0, 1)
			}
			predictData.Set(i, j+1, f)

		}
	}
	return predictData, nil
}
