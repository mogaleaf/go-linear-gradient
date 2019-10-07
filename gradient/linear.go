package gradient

import (
	"go/linear/gradient/cost"
	"log"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func LinearGradient(X mat.Matrix, y mat.Matrix, theta mat.Matrix, alpha float64, num_iters int) (mat.Matrix, error) {
	pts := make(plotter.XYs, 0)
	for i := 0; i < num_iters; i++ {

		r, cx := X.Dims()
		m, _ := y.Dims()
		_, c := theta.Dims()
		H := mat.NewDense(c, r, nil)
		H.Mul(theta.T(), X.T())
		DIFF := mat.NewDense(r, c, nil)
		DIFF.Sub(H.T(), y)
		SUM := mat.NewDense(c, cx, nil)
		SUM.Mul(DIFF.T(), X)

		RESULT := mat.NewDense(cx, c, nil)
		RESULT.Scale(alpha*(1/float64(m)), SUM.T())
		RESULT.Sub(theta, RESULT)

		theta = RESULT

		if i%20 == 0 {
			f, e := cost.ComputeCost(X, y, theta)
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

func show(data plotter.XYs) {
	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "cost function Series"
	p.Y.Label.Text = "cost function value)"
	p.X.Label.Text = "number of iteration"
	p.Add(plotter.NewGrid())

	line, points, err := plotter.NewLinePoints(data)
	if err != nil {
		log.Panic(err)
	}

	p.Add(line, points)
	err = p.Save(10*vg.Centimeter, 5*vg.Centimeter, "cost.png")
	if err != nil {
		log.Panic(err)
	}
}
