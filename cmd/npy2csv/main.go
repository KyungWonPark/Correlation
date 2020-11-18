package main

import (
	"fmt"
	"os"

	"github.com/KyungWonPark/Correlation/internal/io"
)

func main() {
	fileName := os.Args[1]

	npyFile := io.NpytoMat64(fileName)
	fmt.Println("Reading npy file complete")

	io.Mat64toCSV(fileName+".csv", npyFile)

	return
}
