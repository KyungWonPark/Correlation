package io

import (
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
			go parseLine(matrix, parsed, offset, row, &wg)
		}
		wg.Wait()

		for i := 0; i < jobMark; i++ {
			fmt.Fprintf(f, "%s\n", parsed[i])
		}
	}

	return
}

func parseLine(matrix *mat64.Dense, parsed []string, offset int, row int, wg *sync.WaitGroup) {
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
