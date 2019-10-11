package gradient

import (
	"go/linear/gradient/cost"
	"go/linear/gradient/hypothesis"
	"log"

	"golang.org/x/image/colornames"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

//Linear Gradient using slices
func LinearGradient(data [][]float64, y []float64, theta []float64, alpha float64, num_iters int, printCostFunction bool) ([]float64, error) {

	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "Data point"
	p.Y.Label.Text = "cost"
	p.X.Label.Text = "size"
	p.Add(plotter.NewGrid())

	pts := make(plotter.XYs, 0)
	for i := 0; i < num_iters; i++ {
		//Number of training examples
		m := len(y)
		//Slice helper to calculate our new versions of theta
		thetaTemp := make([]float64, len(theta))

		//Sum (hi-yx)xi
		for rowI := 0; rowI < m; rowI++ {
			hi := hypothesis.ComputeHypothesis(data[rowI], theta)
			sumRowI := computeSumRowI(data[rowI], hi, y[rowI])
			for t := 0; t < len(theta); t++ {
				thetaTemp[t] += sumRowI[t]
			}
		}
		//Update theta
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
			if len(thetaTemp) == 2 {
				print2DimData(theta, p)
			}
		}

	}
	if printCostFunction {
		showCost(pts)
	}
	return theta, nil
}

//Multiply by xi only if not theta0
func computeSumRowI(x []float64, hi float64, yi float64) []float64 {
	theta := make([]float64, len(x)+1)
	theta[0] = hi - yi
	for i := 1; i < len(theta); i++ {
		theta[i] = (hi - yi) * x[i-1]
	}
	return theta
}

func print2DimData(theta []float64, p *plot.Plot) {
	pts2 := make(plotter.XYs, 2)
	pts2[0] = plotter.XY{
		X: 0,
		Y: hypothesis.ComputeHypothesis([]float64{0}, theta),
	}
	pts2[1] = plotter.XY{
		X: 2,
		Y: hypothesis.ComputeHypothesis([]float64{2}, theta),
	}
	show(pts2, p)

}

func show(pts2 plotter.XYs, p *plot.Plot) {

	line, _, err := plotter.NewLinePoints(pts2)
	line.Color = colornames.Red
	p.Add(line)

	err = p.Save(10*vg.Centimeter, 5*vg.Centimeter, "size_cost_training.png")
	if err != nil {
		log.Panic(err)
	}
}
