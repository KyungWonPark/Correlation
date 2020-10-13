package anal

import (
	"math"
	"sort"

	"github.com/gonum/matrix/mat64"
)

// Cluster represents cluster configuration
type Cluster [][]int

type tempCluster struct {
	members    map[float64][]int
	cntMembers map[float64]int
}

// GetNonZeroSmallestEigVal returns... as its name says
func GetNonZeroSmallestEigVal(eigVal *mat64.Dense) (float64, int) {
	var nonZeroSmallestEigVal float64
	nonZeroSmallestEigVal = 10
	idxNonZeroSmallestEigVal := -1

	n, _ := eigVal.Dims()
	for i := 0; i < n; i++ {
		val := eigVal.At(i, 0)
		absVal := math.Abs(val)

		if absVal > 0.000000 {
			if absVal < nonZeroSmallestEigVal {
				nonZeroSmallestEigVal = absVal
				idxNonZeroSmallestEigVal = i
			}
		}
	}

	return nonZeroSmallestEigVal, idxNonZeroSmallestEigVal
}

// GetCluster creates cluster from eigenvalue and eigenvectors
func GetCluster(eigVal *mat64.Dense, eigVec *mat64.Dense) Cluster {
	var nonZeroSmallestEigVal float64
	nonZeroSmallestEigVal = 10

	idxNonZeroSmallestEigVal := -1

	n, _ := eigVal.Dims()
	for i := 0; i < n; i++ {
		val := eigVal.At(i, 0)
		absVal := math.Abs(val)

		if absVal > 0.000000 {
			if absVal < nonZeroSmallestEigVal {
				nonZeroSmallestEigVal = absVal
				idxNonZeroSmallestEigVal = i
			}
		}
	}

	var c tempCluster

	rows, _ := eigVec.Dims()
	for i := 0; i < rows; i++ {
		val := eigVec.At(i, idxNonZeroSmallestEigVal)
		c.members[val] = append(c.members[val], i)
		c.cntMembers[val] = c.cntMembers[val] + 1
	}

	var realC Cluster

	for k := range c.members {
		if c.cntMembers[k] >= 2 {
			members := []int{}
			for _, v := range c.members[k] {
				members = append(members, v)
			}

			sort.Ints(members)

			realC = append(realC, members)
		}
	}

	return realC
}

// ANC calculates Average Number of Clusters

// WCC calculates Within Cluster Correlation
