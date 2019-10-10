package learn

import (
	"encoding/csv"
	"go/linear/gradient/gradient"
	"go/linear/gradient/normalize"
	"os"
	"strconv"
)

// Init Matrices with csv file input
func Learn(fileName string, alpha float64, iteration int, printCostFunction bool) ([]float64, []float64, []float64, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, nil, err
	}

	dataLoaded := make([][]float64, len(lines))
	y := make([]float64, len(lines))
	theta := make([]float64, len(lines[0]))
	// Loop through lines & turn into object
	for i, line := range lines {
		dataLoaded[i] = make([]float64, len(line)-1)
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, nil, err
			}

			if err != nil {
				return nil, nil, nil, err
			}
			if j < len(line)-1 {
				dataLoaded[i][j] = f
			} else {
				y[i] = f
			}

		}

	}

	// NormalizeVectorized all the elements to keep an identical scale between different data
	XNorm, M, S, err := normalize.Normalize(dataLoaded)

	// Perform gradient descent to calculate Theta
	THETA, err := gradient.LinearGradient(XNorm, y, theta, alpha, iteration, printCostFunction)
	if err != nil {
		return nil, nil, nil, err
	}

	return THETA, M, S, nil
}
