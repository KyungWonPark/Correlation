package laplacian

// #cgo CFLAGS: -I./files/include -I./files/testing -I./files
// #cgo LDFLAGS: -L./files -llaplacian
// #include <laplacian.h>

import "C"
import (
	"unsafe"

	"github.com/gonum/matrix/mat64"
)

// EigenDecomp carries out eigen decomposition
func EigenDecomp(matrix *mat64.Dense) (*mat64.Dense, *mat64.Dense) {
	rows, cols := matrix.Dims()
	arrMatBuffer := make([]float64, rows*cols)
	matBuffer := mat64.NewDense(rows, cols, arrMatBuffer)
	arrEigVal := make([]float64, rows*1)
	eigVal := mat64.NewDense(rows, 1, arrEigVal)

	pMatBuffer := unsafe.Pointer(&arrMatBuffer[0])
	pEigVal := unsafe.Pointer(&arrEigVal[0])

	C.eigenDecomposition(rows, pMatBuffer, pEigVal)

	return eigVal, matBuffer
}
