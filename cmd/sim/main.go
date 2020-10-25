package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
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
	RESULTDIR := os.Getenv("RESULT")

	c2 := mat64.NewDense(13362, 13362, nil)
	sim := mat64.NewDense(13362, 13362, nil)

	records := readCsvFile(RESULTDIR + "/c2-tilda.csv")
	fmt.Println("Reading CSV complete")

	parseCSV(records, c2)
	fmt.Println("Parsing CSV complete")
	processSim(c2, sim)
	fmt.Println("Processing C2 complete")

	io.Mat64toCSV(RESULTDIR+"/sim.csv", sim)

	return
}

func parseLine(records [][]string, c2 *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	for {
		index, ok := <-order
		if ok {
			for i := 0; i < 13362; i++ {
				value, err := strconv.ParseFloat(records[index][i], 64)
				if err != nil {
					log.Fatalf("Failed to parse: %v\n", err)
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
			for i := 0; i < (index + 1); i++ {
				var accProd float64
				for t := 0; t < 13362; t++ {
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
