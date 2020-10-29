package io

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/gonum/matrix/mat64"
)

// CSVtoMat64 reads CSV file and load them to Mat64
//func CSVtoMat64(path string, matrix *mat64.Dense) {
//}

// Mat64toCSV saves Mat64 as a csv file
func Mat64toCSV(path string, matrix *mat64.Dense) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("[Error] Mat64toCSV: Failed to open: %s\n", path)
	}
	defer f.Close()

	rows, _ := matrix.Dims()

	stride := runtime.NumCPU()
	parsed := make([]string, stride)

	for row := 0; row < rows; row += stride {
		var wg sync.WaitGroup
		jobMark := stride

		if row+stride >= rows {
			jobMark = rows - row
		}

		wg.Add(jobMark)
		for offset := 0; offset < jobMark; offset++ {
			parsed[offset] = ""
			go parseLine0(matrix, parsed, offset, row, &wg)
		}
		wg.Wait()

		for i := 0; i < jobMark; i++ {
			fmt.Fprintf(f, "%s\n", parsed[i])
		}
	}

	return
}

func parseLine0(matrix *mat64.Dense, parsed []string, offset int, row int, wg *sync.WaitGroup) {
	_, cols := matrix.Dims()

	num := ""
	for i := 0; i < cols; i++ {
		num += (strconv.FormatFloat(matrix.At(row+offset, i), 'g', -1, 64) + ", ")
	}

	num2 := strings.TrimSuffix(num, ", ")
	parsed[offset] += num2

	wg.Done()

	return
}

// CSVtoMat64 converts csv file to mat64
func CSVtoMat64(path string, matrix *mat64.Dense) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("Failed to open file: %v\n", err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatalf("Failed to parse CSV file: %v\n", err)
	}

	workers := runtime.NumCPU()
	rows, _ := matrix.Dims()
	order := make(chan int, workers)
	var wg sync.WaitGroup

	wg.Add(rows)

	for i := 0; i < workers; i++ {
		go parseLine1(records, matrix, order, &wg)
	}

	for i := 0; i < rows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)

	return
}

func parseLine1(records [][]string, matrix *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	_, cols := matrix.Dims()

	for {
		index, ok := <-order
		if ok {
			for i := 0; i < cols; i++ {
				str := strings.TrimSpace(records[index][i])
				value, err := strconv.ParseFloat(str, 64)
				if err != nil {
					log.Fatalf("Failed to parse: %v\n", err)
				}

				matrix.Set(index, i, value)
			}

			wg.Done()
		} else {
			break
		}
	}
	return
}
