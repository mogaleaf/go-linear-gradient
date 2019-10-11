package gradient

import (
	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func showCost(data plotter.XYs) {
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
