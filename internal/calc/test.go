package calc

import (
	"math"
	"runtime"
	"sync"

	"github.com/gonum/matrix/mat64"
)

// CheckLaplacian checks whether input matrix satisfies Laplacian matrix or not
func CheckLaplacian(matrix *mat64.Dense, pre float64) bool {
	return SymCheck(matrix, pre) && RowCheck(matrix, pre)
}

// SymCheck checks symmetry
func SymCheck(matrix *mat64.Dense, pre float64) bool {
	rows, _ := matrix.Dims()
	workers := runtime.NumCPU()

	order := make(chan int, workers)
	isSymm := make([]bool, rows)
	var wg sync.WaitGroup

	wg.Add(rows)

	for i := 0; i < workers; i++ {
		go symCheck(matrix, isSymm, math.Abs(pre), order, &wg)
	}

	for i := 0; i < rows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)

	symm := true
	for i := 0; i < rows; i++ {
		symm = symm && isSymm[i]
	}

	return symm
}

func symCheck(matrix *mat64.Dense, isSymm []bool, pre float64, order <-chan int, wg *sync.WaitGroup) {
	_, cols := matrix.Dims()

	for {
		index, ok := <-order
		if ok {
			isSymm[index] = true
			for i := index; i < cols; i++ {
				isSame := (math.Abs(matrix.At(index, i)-matrix.At(i, index)) < pre)
				if !isSame {
					isSymm[index] = false
					break
				}
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// RowCheck checks sum of row elements == 0
func RowCheck(matrix *mat64.Dense, pre float64) bool {
	rows, _ := matrix.Dims()
	workers := runtime.NumCPU()

	order := make(chan int, workers)
	isRowSumZero := make([]bool, rows)
	var wg sync.WaitGroup

	wg.Add(rows)

	for i := 0; i < workers; i++ {
		go rowCheck(matrix, isRowSumZero, math.Abs(pre), order, &wg)
	}

	for i := 0; i < rows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)

	rowZero := true
	for i := 0; i < rows; i++ {
		rowZero = rowZero && isRowSumZero[i]
	}

	return rowZero
}

func rowCheck(matrix *mat64.Dense, isRowSumZero []bool, pre float64, order <-chan int, wg *sync.WaitGroup) {
	_, cols := matrix.Dims()

	for {
		index, ok := <-order
		if ok {
			isRowSumZero[index] = true
			var acc float64

			for i := 0; i < cols; i++ {
				acc = acc + matrix.At(index, i)
			}

			isRowZero := (math.Abs(acc) < pre)
			if !isRowZero {
				isRowSumZero[index] = false
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// CheckEigenQuality check eigenvalue and eigenvector
func CheckEigenQuality(org *mat64.Dense, eigVal *mat64.Dense, eigVec *mat64.Dense, pre float64) bool {
	rows, cols := org.Dims()

	var isReconOK bool
	var isEigvecRight bool

	// Check reconstruction
	{
		eigValMat := mat64.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			eigValMat.Set(i, i, eigVal.At(i, 0))
		}

		// U^T * A
		ua := mat64.NewDense(rows, cols, nil)
		ua.Mul(eigVec, org)

		// S * U^T
		su := mat64.NewDense(rows, cols, nil)
		su.Mul(eigValMat, eigVec)

		isReconOK = mat64.EqualApprox(ua, su, math.Abs(pre))
	}

	// Check eigenvectors are actually eigenvectors
	{
		isEigvecRight = true
		// v
		v := mat64.NewDense(rows, 1, nil)
		// A * v
		av := mat64.NewDense(rows, 1, nil)
		// lambda * v
		lv := mat64.NewDense(rows, 1, nil)
		for i := 0; i < rows; i++ {
			// v init
			for j := 0; j < cols; j++ {
				v.Set(j, 0, eigVec.At(i, j))
			}

			av.Mul(org, v)
			for j := 0; j < cols; j++ {
				val := eigVal.At(i, 0) * v.At(j, 0)
				lv.Set(j, 0, val)
			}

			right := mat64.EqualApprox(av, lv, math.Abs(pre))
			isEigvecRight = isEigvecRight && right
		}
	}

	return isReconOK && isEigvecRight
}
