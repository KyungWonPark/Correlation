package anal

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// Cluster represents cluster configuration
type Cluster map[float64][]int

// GetCluster creates cluster from eigenvalue and eigenvectors
func GetCluster(eigVal *mat64.Dense, eigVec *mat64.Dense) Cluster {
	var nonZeroSmallestEigVal float64
	nonZeroSmallestEigVal = 10

	idxNonZeroSmallestEigVal := -1

	n, _ := eigVal.Dims()
	for i := 0; i < n; i++ {
		val := eigVal.At(i, 1)
		absVal := math.Abs(val)

		if absVal > 0.000000 {
			if absVal < nonZeroSmallestEigVal {
				nonZeroSmallestEigVal = absVal
				idxNonZeroSmallestEigVal = -1
			}
		}
	}

	var clusterConf Cluster

	rows, _ := eigVec.Dims()
	for i := 0; i < rows; i++ {
		val := eigVec.At(i, idxNonZeroSmallestEigVal)
		clusterConf[val] = append(clusterConf[val], i)
	}

	return clusterConf
}

// ANC calculates Average Number of Clusters

// WCC calculates Within Cluster Correlation
