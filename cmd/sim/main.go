package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

func main() {
	c2FileName := os.Args[1]

	c2 := io.NpytoMat64(c2FileName)
	fmt.Println("Reading c2-tilda.npy complete")

	fmt.Printf("c2[0][0]: %f\n", c2.At(0, 0))
	fmt.Printf("c2[1][1]: %f\n", c2.At(1, 1))
	fmt.Printf("c2[2][2]: %f\n", c2.At(2, 2))

	sim := mat64.NewDense(13362, 13362, nil)

	processSim(c2, sim)
	fmt.Println("Processing C2 complete")

	io.Mat64toNpy("similarity-tilda.npy", sim)

	return
}

func parseLine(records [][]string, c2 *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	for {
		index, ok := <-order
		if ok {
			for i := 0; i < 13362; i++ {
				str := strings.TrimSpace(records[index][i])
				value, err := strconv.ParseFloat(str, 64)
				if err != nil {
					log.Fatalf("Failed to parse: %v\n", err)
				}

				if math.Abs(value) > 1 {
					log.Fatalf("Error: value is: %f, larger than 1!\n", value)
				}

				c2.Set(index, i, value)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

func parseCSV(records [][]string, c2 *mat64.Dense) {
	workers := runtime.NumCPU()

	order := make(chan int, workers)
	var wg sync.WaitGroup

	wg.Add(13362)

	for i := 0; i < workers; i++ {
		go parseLine(records, c2, order, &wg)
	}

	for i := 0; i < 13362; i++ {
		order <- i
	}

	wg.Wait()
	close(order)

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
