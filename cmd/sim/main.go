package main

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func main() {
	c2FileName := os.Args[1]

	c2 := io.NpytoMat64(c2FileName)

	sim := mat64.NewDense(13362, 13362, nil)

	processSim(c2, sim)

	io.Mat64toNpy("bin/sim.npy", sim)

	return
}

func processLine(c2 *mat64.Dense, sim *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	for {
		index, ok := <-order
		if ok {
			for i := 0; i < (index + 1); i++ { // for all cols j
				var accProd float64
				for t := 0; t < 13362; t++ { // dotProd(i, j)
					accProd = accProd + (c2.At(index, t) * c2.At(i, t))
				}

				sim.Set(index, i, accProd)
				sim.Set(i, index, accProd)
			}

			fmt.Printf("Processed: Line %d\n", index)

			wg.Done()
		} else {
			break
		}
	}

	return
}

func processSim(c2 *mat64.Dense, sim *mat64.Dense) {
	workers := runtime.NumCPU()

	order := make(chan int, workers)
	var wg sync.WaitGroup

	wg.Add(13362)

	for i := 0; i < workers; i++ {
		go processLine(c2, sim, order, &wg)
	}

	for i := 0; i < 13362; i++ {
		order <- i
	}

	wg.Wait()
	close(order)
	return
}
