package vectorized

import (
	"encoding/csv"
	"go/linear/gradient/gradient"
	"go/linear/gradient/learn"
	"go/linear/gradient/normalize"
	"go/linear/gradient/predict"
	"go/linear/gradient/predict/vectorized"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type learnVectorized struct {
}

func NewlearnVectorized() learn.Learn {
	return &learnVectorized{}
}

// Init Matrices with csv file input
func (s *learnVectorized) Learn(config learn.LearnConfiguration) (predict.Predict, error) {

	X, y, err := loadFileVectorized(config.FileName)
	_, c := X.Dims()
	theta := mat.NewDense(c, 1, nil)

	// NormalizeVectorized all the elements to keep an identical scale between different data
	XNorm, M, S, err := normalize.NormalizeVectorized(X)

	// Perform gradient descent to calculate Theta
	THETA, err := gradient.LinearGradientVectorized(XNorm, y, theta, config.Alpha, config.NumberIteration, config.PrintCostFunction)
	if err != nil {
		return nil, err
	}
	return vectorized.NewPredictVectorized(THETA, M, S), nil
}

func loadFileVectorized(fileName string) (mat.Matrix, mat.Matrix, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	X := mat.NewDense(len(lines), len(lines[0]), nil)
	y := mat.NewDense(len(lines), 1, nil)

	// Loop through lines & turn into object
	for i, line := range lines {
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, err
			}

			if err != nil {
				return nil, nil, err
			}
			if j < len(line)-1 {
				// Already set first column to 1 for theta(0)
				if j == 0 {
					X.Set(i, 0, 1)
				}
				X.Set(i, j+1, f)
			} else {
				y.Set(i, 0, f)
			}

		}

	}
	return X, y, nil
}
