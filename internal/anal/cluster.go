package anal

import (
	"log"
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
	n, _ := eigVal.Dims()

	if n < 2 {
		log.Fatal("[GetNonZeroSmallestEigVal] n is smaller than 2.")
	}

	smallest := 1000000.0
	idxSmallest := -1

	for i := 0; i < n; i++ {
		val := math.Abs(eigVal.At(i, 0))
		if val < smallest {
			smallest = val
			idxSmallest = i
		}
	}

	smallest2nd := 10000000.0
	idxSmallest2nd := -1

	for i := 0; i < n; i++ {
		val := math.Abs(eigVal.At(i, 0))
		if val < smallest2nd && i != idxSmallest {
			smallest2nd = val
			idxSmallest2nd = i
		}
	}

	return smallest2nd, idxSmallest2nd
}

// GetCluster creates cluster from eigenvalue and eigenvectors
func GetCluster(eigVal *mat64.Dense, eigVec *mat64.Dense) Cluster {
	_, idxNonZeroSmallestEigVal := GetNonZeroSmallestEigVal(eigVal)

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