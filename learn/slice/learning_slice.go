package slice

import (
	"encoding/csv"
	"go/linear/gradient/gradient"
	"go/linear/gradient/hypothesis"
	"go/linear/gradient/learn"
	"go/linear/gradient/normalize"
	"go/linear/gradient/predict"
	"go/linear/gradient/predict/slice"
	"log"
	"os"
	"strconv"

	"golang.org/x/image/colornames"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type learnSlice struct {
}

func NewlearnSlice() learn.Learn {
	return &learnSlice{}
}

// Init Slices with csv file input
func (l *learnSlice) Learn(config learn.LearnConfiguration) (predict.Predict, error) {

	inputs, y, err := loadFile(config.TrainingFileName)
	if err != nil {
		return nil, err
	}

	theta := make([]float64, len(inputs[0])+1)

	// Normalize all the elements to keep an identical scale between different data
	XNorm, M, S, err := normalize.Normalize(inputs)

	// Perform gradient descent to calculate Theta
	theta, err = gradient.LinearGradient(XNorm, y, theta, config.Alpha, config.NumberIteration, config.PrintCostFunction)
	if err != nil {
		return nil, err
	}

	if len(inputs[0]) == 1 {
		print2DimData(inputs, y, theta, M, S)
	}

	return slice.NewSlicePredict(config.PredictionFileName, theta, M, S)
}

func loadFile(fileName string) ([][]float64, []float64, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	inputs := make([][]float64, len(lines))
	y := make([]float64, len(lines))

	// Loop through lines & turn into object
	for i, line := range lines {
		inputs[i] = make([]float64, len(line)-1)
		for j, data := range line {
			f, err := strconv.ParseFloat(data, 64)
			if err != nil {
				return nil, nil, err
			}

			if err != nil {
				return nil, nil, err
			}
			if j < len(line)-1 {
				inputs[i][j] = f
			} else {
				y[i] = f
			}

		}

	}
	return inputs, y, nil
}

func print2DimData(inputs [][]float64, y []float64, theta []float64, M []float64, S []float64) {
	pts := make(plotter.XYs, 0)
	for i := 0; i < len(inputs); i++ {
		pts = append(pts, plotter.XY{
			X: inputs[i][0],
			Y: y[i],
		})
	}
	pts2 := make(plotter.XYs, 2)
	pts2[0] = plotter.XY{
		X: 0,
		Y: hypothesis.ComputeHypothesis([]float64{(0 - M[0]) / S[0]}, theta),
	}
	pts2[1] = plotter.XY{
		X: 4000,
		Y: hypothesis.ComputeHypothesis([]float64{(4000 - M[0]) / S[0]}, theta),
	}
	show(pts, pts2)

}

func show(data plotter.XYs, pts2 plotter.XYs) {
	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "Data point"
	p.Y.Label.Text = "cost"
	p.X.Label.Text = "size"
	p.Add(plotter.NewGrid())

	_, points, err := plotter.NewLinePoints(data)
	if err != nil {
		log.Panic(err)
	}

	p.Add(points)

	line, _, err := plotter.NewLinePoints(pts2)
	line.Color = colornames.Red
	p.Add(line)

	err = p.Save(10*vg.Centimeter, 5*vg.Centimeter, "size_cost.png")
	if err != nil {
		log.Panic(err)
	}
}
