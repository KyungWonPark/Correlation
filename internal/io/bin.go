package io

import (
	"encoding/binary"
	"fmt"
	"os"
)

// F64SliceToBin writes float64 slice to a file
func F64SliceToBin(path string, slice []float64) {
	file, err := os.Create(path)
	defer file.Close()
	if err != nil {
		fmt.Printf("[F64SlicetoBin] Failed to create file: %s\n", path)
		return
	}

	binary.Write(file, binary.LittleEndian, slice)
	return
}
