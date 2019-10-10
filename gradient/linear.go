package gradient

import (
	"go/linear/gradient/cost"
	"go/linear/gradient/hypothesis"

	"gonum.org/v1/plot/plotter"
)

func LinearGradient(data [][]float64, y []float64, theta []float64, alpha float64, num_iters int, printCostFunction bool) ([]float64, error) {
	pts := make(plotter.XYs, 0)
	for i := 0; i < num_iters; i++ {

		m := len(y)
		thetaTemp := make([]float64, len(theta))

		for rowI := 0; rowI < m; rowI++ {

			//Sum
			hi := hypothesis.ComputeHypothesis(data[rowI], theta)
			sumRowI := computeSumRowI(data[rowI], hi, y[rowI])

			for t := 0; t < len(theta); t++ {
				thetaTemp[t] += sumRowI[t]
			}

		}
		for t := 0; t < len(theta); t++ {
			theta[t] = theta[t] - (alpha/float64(m))*thetaTemp[t]
		}

		if printCostFunction && i%20 == 0 {
			f, e := cost.ComputeCost(data, y, theta)
			if e != nil {
				return nil, e
			}
			pts = append(pts, plotter.XY{
				X: float64(i),
				Y: f,
			})
		}

	}
	show(pts)
	return theta, nil
}

func computeSumRowI(x []float64, hi float64, yi float64) []float64 {
	theta := make([]float64, len(x)+1)
	theta[0] = hi - yi
	for i := 1; i < len(theta); i++ {
		theta[i] = (hi - yi) * x[i-1]
	}
	return theta
}
