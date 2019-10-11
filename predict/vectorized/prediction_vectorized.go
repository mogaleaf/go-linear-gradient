package vectorized

import (
	"encoding/csv"
	"fmt"
	"go/linear/gradient/hypothesis"
	"go/linear/gradient/predict"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type predictVectorized struct {
	theta mat.Matrix
	M     mat.Matrix
	S     mat.Matrix
}

func NewPredictVectorized(theta mat.Matrix, M mat.Matrix, S mat.Matrix) predict.Predict {
	return &predictVectorized{
		S:     S,
		M:     M,
		theta: theta,
	}
}

func (p *predictVectorized) Predict(predictionFile string, resultFile string) error {
	f, err := os.Open(predictionFile)
	if err != nil {
		return err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return err
	}

	predictData := mat.NewDense(len(lines), len(lines[0])+1, nil)
	for i, line := range lines {
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return err
			}

			if err != nil {
				return err
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
		return v - p.M.At(0, j-1)
	}, predictData)

	predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v / p.S.At(0, j-1)
	}, predictData)

	// Calc h(x)
	result, err := hypothesis.ComputeHypothesisVectorized(p.theta, predictData)
	if err != nil {
		return err
	}

	return writeDataVectorized(result, lines, resultFile)
}

func writeDataVectorized(resultMat mat.Matrix, lines [][]string, resultFile string) error {
	f, err := os.Create(resultFile)
	if err != nil {
		return err
	}
	defer f.Close()
	writer := csv.NewWriter(f)
	r, c := resultMat.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			newLine := append(lines[j], fmt.Sprintf("%0.10f", resultMat.At(i, j)))
			writer.Write(newLine)
			println(fmt.Sprintf("prediction %s = %f", lines[j], resultMat.At(i, j)))
		}

	}
	writer.Flush()
	return nil
}
