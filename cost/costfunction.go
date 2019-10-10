package cost

import (
	"go/linear/gradient/hypothesis"
)

func ComputeCost(data [][]float64, y []float64, theta []float64) (float64, error) {
	m := len(y)
	sum := 0.0
	for rowI := 0; rowI < m; rowI++ {

		//Sum
		hi := hypothesis.ComputeHypothesis(data[rowI], theta)
		sum += (hi - y[rowI]) * (hi - y[rowI])

	}
	return (1 / float64(2*m)) * sum, nil
}
