package learn

import (
	"encoding/csv"
	"go/linear/gradient/gradient"
	"go/linear/gradient/normalize"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func Learn(fileName string, readInteger bool, alpha float64, iteration int, printCostFunction bool) (mat.Matrix, mat.Matrix, mat.Matrix, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, nil, err
	}

	dataLoaded := make([][]int64, len(lines))
	X := mat.NewDense(len(lines), len(lines[0]), nil)
	y := mat.NewDense(len(lines), 1, nil)
	theta := mat.NewDense(len(lines[0]), 1, nil)
	// Loop through lines & turn into object
	for i, line := range lines {
		dataLoaded[i] = make([]int64, len(line))
		for j, data := range line {
			var f float64
			if readInteger {
				rI, err := strconv.ParseInt(data, 10, 64)
				if err != nil {
					return nil, nil, nil, err
				}
				f = float64(rI)
			} else {
				f, err = strconv.ParseFloat(data, 64)
				if err != nil {
					return nil, nil, nil, err
				}
			}
			if err != nil {
				return nil, nil, nil, err
			}
			if j < len(line)-1 {
				if j == 0 {
					X.Set(i, 0, 1)
				}
				X.Set(i, j+1, f)
			} else {
				y.Set(i, 0, f)
			}

		}

	}
	XNorm, M, S, err := normalize.Normalize(X)

	matrix, err := gradient.LinearGradient(XNorm, y, theta, alpha, iteration, printCostFunction)
	if err != nil {
		return nil, nil, nil, err
	}
	return matrix, M, S, nil
}
