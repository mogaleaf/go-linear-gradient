package gradient

import (
	"go/linear/gradient/cost"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/plotter"
)

//Linear Gradient using Matrices
func LinearGradientVectorized(X mat.Matrix, y mat.Matrix, theta mat.Matrix, alpha float64, num_iters int, printCostFunction bool) (mat.Matrix, error) {
	pts := make(plotter.XYs, 0)
	for i := 0; i < num_iters; i++ {

		r, cx := X.Dims()
		m, _ := y.Dims()
		_, c := theta.Dims()
		H := mat.NewDense(c, r, nil)
		// Calculate h(x)
		H.Mul(theta.T(), X.T())
		// Calculate h(x)-y
		DIFF := mat.NewDense(r, c, nil)
		DIFF.Sub(H.T(), y)
		// Calculate Sum(1,m)(h(x)-y)x
		SUM := mat.NewDense(c, cx, nil)
		SUM.Mul(DIFF.T(), X)

		RESULT := mat.NewDense(cx, c, nil)
		// Apply grad step alpha
		RESULT.Scale(alpha*(1/float64(m)), SUM.T())
		RESULT.Sub(theta, RESULT)

		// Update theta and start again
		theta = RESULT

		if printCostFunction && i%20 == 0 {
			f, e := cost.ComputeCostvectorized(X, y, theta)
			if e != nil {
				return nil, e
			}
			pts = append(pts, plotter.XY{
				X: float64(i),
				Y: f,
			})
		}

	}
	if printCostFunction {
		showCost(pts)
	}
	return theta, nil
}
