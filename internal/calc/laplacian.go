package calc

import (
	"sync"

	"github.com/gonum/matrix/mat64"
)

func laplacian(inputMat *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	_, inputCols := inputMat.Dims()

	for {
		index, ok := <-order
		if ok {
			var degree float64

			for i := 0; i < inputCols; i++ {
				degree += inputMat.At(index, i)
			}

			for i := 0; i < inputCols; i++ {
				value := inputMat.At(index, i)
				inputMat.Set(index, i, -value)
			}

			value := inputMat.At(index, index)
			value += degree
			inputMat.Set(index, index, value)

			wg.Done()
		} else {
			break
		}
	}

	return
}

// Laplacian turns an adjacency matrix into Laplacian matrix
func (p *PipeLine) Laplacian(inputMat *mat64.Dense) {
	inputRows, _ := inputMat.Dims()

	order := make(chan int, p.numPoper)
	var wg sync.WaitGroup

	wg.Add(inputRows)

	for i := 0; i < p.numPoper; i++ {
		go laplacian(inputMat, order, &wg)
	}

	for i := 0; i < inputRows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)
	return
}
