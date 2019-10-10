package learn

import (
	"encoding/csv"
	"go/linear/gradient/gradient"
	"go/linear/gradient/normalize"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// Init Matrices with csv file input
func LearnVectorized(fileName string, alpha float64, iteration int, printCostFunction bool) (mat.Matrix, mat.Matrix, mat.Matrix, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, nil, err
	}

	X := mat.NewDense(len(lines), len(lines[0]), nil)
	y := mat.NewDense(len(lines), 1, nil)
	theta := mat.NewDense(len(lines[0]), 1, nil)
	// Loop through lines & turn into object
	for i, line := range lines {
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, nil, err
			}

			if err != nil {
				return nil, nil, nil, err
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

	// NormalizeVectorized all the elements to keep an identical scale between different data
	XNorm, M, S, err := normalize.NormalizeVectorized(X)

	// Perform gradient descent to calculate Theta
	THETA, err := gradient.LinearGradientVectorized(XNorm, y, theta, alpha, iteration, printCostFunction)
	if err != nil {
		return nil, nil, nil, err
	}
	return THETA, M, S, nil
}
